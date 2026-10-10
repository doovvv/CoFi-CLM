"""Global energy-ratio NMSE for validation and the paper's test protocol."""

import math


class NMSE:
    def __init__(self, center=0.5):
        self.center = center
        self.error = 0.0
        self.power = 0.0
        self.samples = 0

    def update(self, target, reconstruction):
        target, reconstruction = target.float() - self.center, reconstruction.float() - self.center
        self.error += ((target - reconstruction) ** 2).sum().item()
        self.power += (target**2).sum().item()
        self.samples += len(target)

    def db(self):
        if self.samples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        return 10 * math.log10(self.error / (self.power + 1e-8) + 1e-8)
