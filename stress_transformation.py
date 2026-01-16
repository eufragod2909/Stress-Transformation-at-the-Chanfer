import pickle
import numpy as np
import math
import json


class StressTransformer:
    def __init__(self, interpolator_path="backend/data/interpolators.pkl"):
        self.interpolator_path = interpolator_path
        self.interpolators = self._load_interpolators()

    def _load_interpolators(self):
        with open(self.interpolator_path, "rb") as f:
            interpolators = pickle.load(f, encoding='latin1') 

        if len(interpolators) != 3:
            print(f"[Warning] Expected 3 interpolators, found: {len(interpolators)}")

        return interpolators

    def get_interpolated_stress(self, x, z):
        s11 = self.interpolators[0](x, z)
        s33 = self.interpolators[1](x, z)
        s13 = self.interpolators[2](x, z)

        return float(s11), float(s33), float(s13)

    def transform_stress(self, sigma_xx, sigma_zz, tau_xz, theta_degrees):
        theta_rad = math.radians(theta_degrees)
        sin_2theta = math.sin(2 * theta_rad)
        cos_2theta = math.cos(2 * theta_rad)

        sigma_avg = (sigma_xx + sigma_zz) / 2.0
        sigma_diff = (sigma_xx - sigma_zz) / 2.0

        sigma_xx_prime = sigma_avg + sigma_diff * cos_2theta + tau_xz * sin_2theta
        sigma_zz_prime = sigma_avg - sigma_diff * cos_2theta - tau_xz * sin_2theta
        tau_xz_prime = -sigma_diff * sin_2theta + tau_xz * cos_2theta

        return {
            "angle": theta_degrees,
            "sigma_xx_prime": sigma_xx_prime,
            "sigma_zz_prime": sigma_zz_prime,
            "tau_xz_prime": tau_xz_prime
        }


if __name__ == "__main__":
    print("\n=== STARTING STRESS TRANSFORMER TEST ===\n")
    transformer = StressTransformer()

    with open('backend/data/cpress_nodes.json', "r") as f:
        cpress_data = json.load(f)
    
    angle = 0.0
    for n, ele in enumerate(cpress_data):
        coords = ele['coordinate']
        s11_original, s33_original, s13_original = transformer.get_interpolated_stress(coords[0], coords[2])

        
        transformed_stresses = transformer.transform_stress(s11_original, s33_original, s13_original, angle)
        sigma_xx_prime = transformed_stresses['sigma_xx_prime']
        cpress = ele['CPRESS']

        print(f"Node {n}: sigma_xx_prime = {sigma_xx_prime:.3f}, cpress = {cpress:.3f}")
