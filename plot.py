import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class StressPlotter:
    def __init__(self, stress_data_path, cpress_data_path):
        self.stress_data_path = stress_data_path
        self.cpress_data_path = cpress_data_path

    def load_cpress_nodes(self):
        with open(self.cpress_data_path, 'r') as file:
            data_cpress = json.load(file)
        return data_cpress

    def load_stress_data(self, name_job):
        with open(self.stress_data_path, 'r') as file:
            data = json.load(file)
        
        elements = data[name_job]['elements_zoi']
        
        x_coords = []
        z_coords = []
        s11_list = []
        s33_list = []
        s13_list = []

        for key, element in elements.items():
            centroid = element['centroid']
            stress = element['data_stress']

            x_coords.append(centroid[0])
            z_coords.append(centroid[1])
            
            s11_list.append(stress[0])
            s33_list.append(stress[2])
            s13_list.append(stress[4])

        return (
            np.array(x_coords),
            np.array(z_coords),
            np.array(s11_list),
            np.array(s33_list),
            np.array(s13_list)
        )

    def transform_stress(self, s11, s33, s13, theta_degrees):
        theta_rad = np.radians(theta_degrees)
        sin_2theta = np.sin(2 * theta_rad)
        cos_2theta = np.cos(2 * theta_rad)

        sigma_avg = (s11 + s33) / 2.0
        sigma_diff = (s11 - s33) / 2.0

        sigma_xx_prime = sigma_avg + sigma_diff * cos_2theta + s13 * sin_2theta
        sigma_zz_prime = sigma_avg - sigma_diff * cos_2theta - s13 * sin_2theta
        tau_xz_prime = -sigma_diff * sin_2theta + s13 * cos_2theta

        return sigma_xx_prime, sigma_zz_prime, tau_xz_prime

    def plot_field(self, x, z, stress_values, component_name="Sigma XX Prime", cpress_nodes=None):
        triang = tri.Triangulation(x, z)
        
        v_min = np.min(stress_values)
        v_max = 0.0
        levels = np.linspace(v_min, v_max, 20)

        plt.figure(figsize=(10, 6))
        
        contour = plt.tricontourf(
            triang, 
            stress_values, 
            levels=levels, 
            cmap='jet'
        )
        
        cbar = plt.colorbar(contour)
        cbar.set_label(f'Stress: {component_name} [MPa]')
        cbar.set_ticks(levels)

        if cpress_nodes:
            cp_x = [node['coordinate'][0] for node in cpress_nodes]
            cp_z = [node['coordinate'][2] for node in cpress_nodes]
            cp_vals = [node['CPRESS'] for node in cpress_nodes]
            
            plt.scatter(
                cp_x, cp_z, 
                color='white', 
                edgecolors='black', 
                marker='o', 
                s=50, 
                label='CPRESS Nodes',
                zorder=10
            )
            
            for i, val in enumerate(cp_vals):
                plt.annotate(
                    f"ID: {i+1}, CPRESS: {val:.2f}", 
                    (cp_x[i], cp_z[i]),
                    textcoords="offset points", 
                    xytext=(5, 5), 
                    ha='left',
                    fontsize=9,
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )

            plt.legend(loc='upper right')
        
        plt.title(f'Transformed Stress Distribution: {component_name}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    stress_json_path = "backend/data/data.json"
    cpress_json_path = "backend/data/cpress_nodes.json"
    
    plotter = StressPlotter(stress_json_path, cpress_json_path)

    x, z, s11, s33, s13 = plotter.load_stress_data("SABOR")
    
    cpress_points = plotter.load_cpress_nodes()

    angle = -20.0
    s_xx_p, s_zz_p, t_xz_p = plotter.transform_stress(s11, s33, s13, angle)

    plotter.plot_field(
        x, z, 
        s_xx_p, 
        component_name="Sigma XX Prime", 
        cpress_nodes=cpress_points
    )