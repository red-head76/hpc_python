import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt


chess_board = (np.arange(64).reshape(8, 8) + (np.arange(64).reshape(8, 8) % 16 >= 8)) % 2
print(chess_board)

a = np.arange(4)
b = 10 * a
b.shape = (4, 1)
print(a + b)

a = np.array([[12, 16],
              [8, 12]])
b = np.array([4, 0])
print(np.linalg.solve(a, b))

sigma = np.array([[15, -35],
                  [-35, 15]])
print(np.linalg.eigh(sigma))


def pendulum(x):
    return x - 2 * np.cos(x)


solution = root(pendulum, x0=-1.5).x
print(solution)
