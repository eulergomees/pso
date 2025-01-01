import time
import numpy as np
import matplotlib.pyplot as plt
import random

maxit = 20
w = 0.5
c1 = 1.5
c2 = 1.5
particles = 100
dim = 2
grid = [-5.12, 5.12]

def rastringin(position):
    n = len(position)
    base_fitness = 10 * n + np.sum(position**2 - 10 * np.cos(2 * np.pi * position))

    g = np.sin(2 * np.pi * position) + 0.5
    h = np.abs(np.cos(2 * np.pi * position) + 0.5)

    penalty = np.sum(np.maximum(0, g)) + np.sum(h)
    return base_fitness + 1e6 * penalty

positions = np.random.uniform(-5.12, 5.12, (particles, dim))
velocities = np.random.uniform(-1, 1, (particles, dim))

pbest_pos = np.copy(positions)
pbest_fit = np.array([rastringin(p) for p in positions])
gbest_pos = positions[np.argmin(pbest_fit)]  # Verifica a melhor posição
gbest_fit = np.min(pbest_fit)

ftmedia = []
ftgbest = []

#PSO
start_time = time.time()
for i in range(maxit):
    for j in range(particles):
        inertia = w * velocities[j]
        cognitive = c1 * np.random.uniform(0, 1, dim) * (pbest_pos[j] - positions[j])
        social = c2 * np.random.uniform(0, 1, dim) * (gbest_pos - positions[j])

        velocities[i] = inertia + cognitive + social
        positions[i] += velocities[i]

        positions[i] = np.clip(positions[i], grid[0], grid[1])




end = start_time - time.time()