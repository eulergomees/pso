import time
import numpy as np
import matplotlib.pyplot as plt
import random

maxit = 20
w = 0.5 #Inertia
c1 = 1.5 #Cog
c2 = 1. #Social
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

def rosenbrock(position):
    n = len(position)
    base_fitness = 100 * (position[1] - position[0]**2)**2 + (1 - position[0])**2

    return base_fitness

#Random positions and velocities for each particle
positions = np.random.uniform(-5.12, 5.12, (particles, dim))
velocities = np.random.uniform(-1, 1, (particles, dim))


pbest_positions = np.copy(positions)
pbest_fitness = np.array([rastringin(p) for p in positions])
gbest_position = positions[np.argmin(pbest_fitness)]
gbest_fitness = np.min(pbest_fitness)

ftmean = []
ftgbest = []
ftstd = []

#PSO
start_time = time.time()
for it in range(maxit):
    for i in range(particles):
        #Calculate inertia, cognitive and social for each particle
        inertia = w * velocities[i]
        cognitive = c1 * np.random.uniform(0, 1, dim) * (pbest_positions[i] - positions[i])
        social = c2 * np.random.uniform(0, 1, dim) * (gbest_position - positions[i])

        #Update the velocite and position
        velocities[i] = inertia + cognitive + social
        positions[i] += velocities[i]

        #Limit for search space
        positions[i] = np.clip(positions[i], grid[0], grid[1])

        #Check the new fitness
        current_fitness = rastringin(positions[i]) #Fitness function (rastringin, rosenbrock,...)

        #Update pbest
        if current_fitness < pbest_fitness[i]:
            pbest_positions[i] = np.copy(positions[i])
            pbest_fitness[i] = current_fitness

            #Update gbest
            if current_fitness < gbest_fitness:
                gbest_pos = np.copy(positions[i])
                gbest_fit = current_fitness

    mean_fitness = np.mean(pbest_fitness) #Calculate the mean fit for all particles in current iter
    std_fitness = np.std(pbest_fitness) #Calculate the std deviation for all particles in current iter
    ftmean.append(mean_fitness) #Add the calculated mean fit in ftmean vector
    ftgbest.append(gbest_fitness) #Add the gbest fit in ftgbest vector
    ftstd.append(std_fitness) #Add the sta deviation in ftstd vector

#Results
execution_time = time.time() - start_time
print(f"Tempo de execução: {execution_time:.4f} segundos")
print(f"Posição ótima encontrada (gbest): {gbest_position}")
print(f"Fitness do gbest: {gbest_fitness}")