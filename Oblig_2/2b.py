# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:45:27 2019

@author: paul
"""

import matplotlib.pyplot as plt
import numpy as np
import math

u = [1,2]

# Hard coded dot product (and transpose) function for (2x2) x (2x1) = (2x1).
def dot_product(A, u):
    v = [0]*len(u)
    
    v[0] = A[0][0] * u[0] + A[0][1] * u[1]
    v[1] = A[1][0] * u[0] + A[1][1] * u[1]
    
    return v

# Returns a matrix based on the given trigonometric variables.
def matrix(theta):
    T = [[math.cos(theta),   math.sin(theta)],
            [-math.sin(theta),  math.cos(theta)]]
    return T


# Set steps, make x (linspace) and empty y-list
steps = 100
x = np.linspace(0, 2*math.pi, steps)
y = [None]*steps

# Loop the function through all theta, dot_product(matrix(theta), u).
for idx in range(steps):
    theta = x[idx]
    T = matrix(theta)
    y[idx] = dot_product(T, u)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
plt.plot(x, y)
fig.show()