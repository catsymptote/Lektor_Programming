#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:53:12 2019

@author: catsymptote
"""

import matplotlib.pylab as plt
import numpy as np

# Initiate matrix A (2x2), and vector/point u (2x1)
A = np.array([[1, 2], [3, 4]])
u = np.array([5, 2])

# Hardcoded (2x2) x (2x1) matrix-vector dot-product function.
def dot_product(A, u):
    dot = np.array([0, 0])
    dot[0] = A[0,0] * u[0] + A[0, 1] * u[1]
    dot[1] = A[1,0] * u[0] + A[1, 1] * u[1]
    return dot

v = dot_product(A, u)
print(v)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
plt.plot(v[0], v[1], 'ro')

# Points suck.
x_min = 0
x_max = 0
y_min = 0
y_max = 0
if v[0] < 0:
    x_min = 1.2 * v[0]
    x_max = 0
else:
    x_min = 0
    x_max = 1.2 * v[0]
if v[1] < 0:
    y_min = 1.2 * v[1]
    y_max = 0
else:
    y_min = 0
    y_max = 1.2 * v[1]
plt.axis([x_min, x_max, y_min, y_max])

fig.show()
