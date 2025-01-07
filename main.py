import time
import numpy as np
import matplotlib.pyplot as plt

maxit = 100
w = 0.5  # Inertia
c1 = 1.5  # Cog
c2 = 1.5  # Social
particles = 30
dim = 5
grid = [-5.12, 5.12]


def rastringin(position):
    n = len(position)
    base_fitness = 10 * n + np.sum(position ** 2 - 10 * np.cos(2 * np.pi * position))

    g = np.sin(2 * np.pi * position) + 0.5
    h = np.abs(np.cos(2 * np.pi * position) + 0.5)

    penalty = np.sum(np.maximum(0, g)) + np.sum(h)
    return base_fitness + 1e6 * penalty


# Random positions and velocities for each particle
positions = np.random.uniform(grid[0], grid[1], (particles, dim))
velocities = np.random.uniform(-1, 1, (particles, dim))

pbest_positions = np.copy(positions)
pbest_fitness = np.apply_along_axis(rastringin, 1, positions)
gbest_index = np.argmin(pbest_fitness)
gbest_position = positions[gbest_index]
gbest_fitness = pbest_fitness[gbest_index]

# Metrics
ftmedia = []
ftgbest = []
ftstd = []

# PSO
start_time = time.time()
for it in range(maxit):
    # Calculate inertia, cognitive and social for each particle
    r1, r2 = np.random.uniform(0, 1, (2, particles, dim))
    inertia = w * velocities
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)

    # Update the velocite and position
    velocities = inertia + cognitive + social
    positions += velocities
    positions = np.clip(positions, grid[0], grid[1])

    # Fitness check
    fitness_values = np.apply_along_axis(rastringin, 1, positions)

    # Update pbest for particles with better fitness
    better_mask = fitness_values < pbest_fitness  # Identify particles with better fitness
    pbest_positions[better_mask] = positions[better_mask]  # Update their positions
    pbest_fitness[better_mask] = fitness_values[better_mask]  # Update their fitness

    # Update gbest if a new global best is found
    current_gbest_index = np.argmin(pbest_fitness)  # Find the best particle
    if pbest_fitness[current_gbest_index] < gbest_fitness:  # Check if it improves gbest
        gbest_position = pbest_positions[current_gbest_index]  # Update gbest position
        gbest_fitness = pbest_fitness[current_gbest_index]  # Update gbest fitness

    # Collect metrics for this iteration
    ftmedia.append(np.mean(pbest_fitness))  # Mean fitness of all particles
    ftgbest.append(gbest_fitness)  # Fitness of the global best particle
    ftstd.append(np.std(pbest_fitness))  # Standard deviation of particle fitness

# Results
execution_time = time.time() - start_time
print(f"Tempo de execução: {execution_time:.4f} segundos")
print(f"Posição ótima encontrada (gbest): {gbest_position}")
print(f"Fitness do gbest: {gbest_fitness}")

# Add plots
plt.figure(figsize=(18, 6))

# Average fit chart
plt.subplot(1, 3, 1)
plt.plot(ftmedia, label='Fitness Média')
plt.xlabel('Iteração')
plt.ylabel('Fitness Média')
plt.legend()

# Gbest fitness chart
plt.subplot(1, 3, 2)
plt.plot(ftgbest, label='Fitness do gbest', color='red')
plt.xlabel('Iteração')
plt.ylabel('Fitness do gbest')
plt.legend()

# Std chart
plt.subplot(1, 3, 3)
plt.plot(ftstd, label='Desvio Padrão da Fitness', color='green')
plt.xlabel('Iteração')
plt.ylabel('Desvio Padrão')
plt.legend()

# Plot
plt.tight_layout()
plt.show()
