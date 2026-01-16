import json
import os
import math
import pickle
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from odbAccess import openOdb
from abaqusConstants import POINT_LIST
from abaqus import session


class OdbDataExtractor:
    LOG_FILE = "log/abaqus_log.txt"
    FULL_FRACTION_LIMIT = 0.999
    TOLERANCE_FACTOR = 0.5
    ZOI_MIN_NODES = 8

    def __init__(self, config_odb, backend_project_path):
        self.config_odb = config_odb
        self.backend_project_path = backend_project_path
        self.output_dir = os.path.join(
            os.path.dirname(self.backend_project_path),
            "backend/data"
        )
        self.extracted_data = {}
        
        self.nodes_in_zoi = {}
        self.node_labels_in_zoi = set()
        self.zoi_element_labels = set()
        self.stress_map = {}
        
        self.interpolation_points = []
        self.interpolation_values = []
        self.interpolator = None

    def run(self):
        for odb_name, odb_config in self.config_odb.items():
            self._process_odb(odb_name, odb_config)
        
        self._save_data()
        self._save_interpolators()

    def _process_odb(self, odb_name, odb_config):
        self.current_odb_name = str(odb_name)
        self.current_config = odb_config
        self.instance_name = str(self.current_config["instance_name"])
        self.field_basename = str(self.current_config["field_basename"])
        odb_path = str(self.current_config['odb_path'])

        self._log("[Extractor] Extraction of ODB: {}".format(self.current_odb_name))
        self._log("  [Extraction] Opening ODB: {}".format(odb_path))

        try:
            odb = openOdb(path=odb_path)
            self._initialize_dataset()

            self._map_stress_data(odb)
            self._filter_nodes(odb)
            self._filter_elements(odb)
            
            self._collect_interpolation_points()
            self._build_interpolator()
            
        except Exception as e:
            self._log("  [Error] Failed to process {}: {}".format(self.current_odb_name, e))
            raise

    def _initialize_dataset(self):
        self.extracted_data[self.current_odb_name] = {
            'elements_zoi': {},
            'nodes_zoi': {}
        }

    def _log(self, msg):
        log_dir = os.path.dirname(self.LOG_FILE)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(self.LOG_FILE, "a") as f:
            f.write("{}\n".format(msg))

    def _map_stress_data(self, odb):
            step_name = str(self.current_config["step_name"])
            frame_target = self.current_config["frame_target"]
            step = odb.steps[step_name]
            frame = step.frames[frame_target]

            field_name = "S_{}".format(self.field_basename)

            field_output = frame.fieldOutputs[field_name]
            self.stress_map = {}
            
            for val in field_output.values:
                if val.instance.name == self.instance_name:
                    self.stress_map[val.elementLabel] = tuple(float(x) for x in val.data)

    def _filter_nodes(self, odb):
        instance = odb.rootAssembly.instances[self.instance_name]
        zoi = self.current_config["zoi_coordinates"]
        tolerance = zoi['ele_size'] * self.TOLERANCE_FACTOR

        x_sorted = sorted([zoi['x1'], zoi['x2']])
        y_sorted = sorted([zoi['y1'], zoi['y2']])
        z_sorted = sorted([zoi['z1'], zoi['z2']])

        min_x, max_x = x_sorted[0] - tolerance, x_sorted[1] + tolerance
        min_y, max_y = y_sorted[0] - tolerance, y_sorted[1] + tolerance
        min_z, max_z = z_sorted[0] - tolerance, z_sorted[1] + tolerance

        self.nodes_in_zoi = {}
        self.node_labels_in_zoi = set()
        nodes_dataset = self.extracted_data[self.current_odb_name]['nodes_zoi']

        for node in instance.nodes:
            coords = node.coordinates
            if min_x <= coords[0] <= max_x and min_y <= coords[1] <= max_y and min_z <= coords[2] <= max_z:
                node_coords = {'coords': tuple(float(c) for c in coords)}
                self.nodes_in_zoi[node.label] = node_coords
                self.node_labels_in_zoi.add(node.label)
                nodes_dataset[str(node.label)] = node_coords

    def _filter_elements(self, odb):
        instance = odb.rootAssembly.instances[self.instance_name]
        self.zoi_element_labels = set()

        for element in instance.elements:
            valid_nodes = [
                n for n in element.connectivity if n in self.node_labels_in_zoi
            ]

            is_inside_zoi = len(valid_nodes) >= self.ZOI_MIN_NODES

            if is_inside_zoi:
                data_stress = self.stress_map[element.label]
                self._process_valid_element(element, valid_nodes, data_stress)
                self.zoi_element_labels.add(element.label)

    def _process_valid_element(self, element, valid_nodes, data_stress):
        str_label = str(element.label)

        connectivity_data = []
        for node_id in valid_nodes:
            connectivity_data.append({
                'nid': node_id,
                'coord': self.nodes_in_zoi[node_id]['coords']
            })

        x_sum = sum(d['coord'][0] for d in connectivity_data)
        z_sum = sum(d['coord'][2] for d in connectivity_data)
        count = float(len(connectivity_data))
        centroid = (x_sum / count, z_sum / count)

        element_data = {
            'connectivity': connectivity_data,
            'centroid': centroid,
            'data_stress': data_stress
        }

        elements_zoi = self.extracted_data[self.current_odb_name]['elements_zoi']
        elements_zoi[str_label] = element_data

    def _collect_interpolation_points(self):
        self.interpolation_points = []
        self.interpolation_values = []
        
        dataset = self.extracted_data[self.current_odb_name]['elements_zoi']
        
        for _, ele_data in dataset.items():
            centroid = ele_data['centroid']

            s11 = ele_data['data_stress'][0]
            s33 = ele_data['data_stress'][2]
            s13 = ele_data['data_stress'][4]
            
            current_val = [s11, s33, s13]

            self.interpolation_points.append(centroid)
            self.interpolation_values.append(current_val)

    def _build_interpolator(self):
        if not self.interpolation_points:
            self._log("  [Warning] No points found for interpolation.")
            return

        points_array = np.array(self.interpolation_points)
        values_array = np.array(self.interpolation_values)
        
        self.interpolators = []

        try:
            num_components = values_array.shape[1]
            for i in range(num_components):
                component_values = values_array[:, i]
                interp = LinearNDInterpolator(points_array, component_values)
                self.interpolators.append(interp)

            self._log("  [Interpolator] Built {} linear interpolation functions successfully.".format(len(self.interpolators)))
        except Exception as e:
            self._log("  [Error] Failed to build interpolators: {}".format(e))

    def _save_interpolators(self):
        if hasattr(self, 'interpolators') and self.interpolators:
            file_path = os.path.join(self.output_dir, "interpolators.pkl")
            try:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                with open(file_path, "wb") as f:
                    pickle.dump(self.interpolators, f)
                self._log("  [Interpolator] Functions saved to: {}".format(file_path))
            except Exception as e:
                self._log("  [Error] Failed to save interpolators: {}".format(e))
        else:
            self._log("  [Warning] No interpolators found to save.")

    def _save_data(self):
        self._log("[Extractor] Saving all extracted data to JSON file...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_path = os.path.join(self.output_dir, "data.json")

        with open(output_path, "w") as f:
            json.dump(self.extracted_data, f, indent=4)

        self._log("  [Extraction] File saved to: {}".format(output_path))