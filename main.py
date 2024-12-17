import time
import numpy as np

maxit = 20
w = 0.5
c1 = 1.5
c2 = 1.5
particles = 100
dimension = 2


def rastringin(x):
    n = len(x)
    return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])
