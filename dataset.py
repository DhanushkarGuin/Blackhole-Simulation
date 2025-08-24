import csv
import numpy as np
from simulation2D import c, G, BlackHole, Ray

SagA = BlackHole((0.0, 0.0, 0.0), 8.54e36)

rays = []
x0 = -1e11
for y0 in np.linspace(-5e10, 5e10, 10):
    rays.append(Ray((x0, y0), (c, 0.0)))

with open("ray_dataset.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ray_id", "step", "x", "y", "r", "phi", "dr", "dphi", "E", "L"])

    for step in range(1000):
        for i, ray in enumerate(rays):
            ray.step(0.01, SagA.r_s)
            writer.writerow([i, step, ray.x, ray.y, ray.r, ray.phi, ray.dr, ray.dphi, ray.E, ray.L])