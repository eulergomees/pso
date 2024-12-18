import time
import numpy as np
import matplotlib.pyplot as plt
import random

maxit = 20
w = 0.5
c1 = 1.5
c2 = 1.5
particles = 100
dimension = 2


def rastringin(x):
    n = len(x)
    return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])


positions = np.random.uniform(-5.12, 5.12, (particles, dimension))
velocities = np.random.uniform(-1, 1, (particles, dimension))

pbest_pos = np.copy(positions)
pbest_fit = np.array([rastringin(p) for p in positions])
gbest_pos = positions[np.argmin(pbest_fit)]  # Verifica a melhor posição
gbest_fit = np.min(pbest_fit)

ftmedia = []
ftgbest = []

for i in range(maxit):
    for j in range(particles):
        inertia = w * velocities[j]
        cognitive = c1 * np.random.uniform(0, 1, dimension) * (pbest_pos[j] - positions[j])
        social = c2 * np.random.uniform(0, 1, dimension) * (gbest_pos - positions[j])