#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:37:28 2019

@author: catsymptote
"""

import matplotlib.pylab as plt
import numpy as np
import math
from transpose import transpose

rect = [
        [1, 1, 3, 3],
        [1, 3, 1, 3]
        ]

new_rect = transpose(rect)

print(new_rect)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
#plt.plot(rect)



theta = 1#2 * math.pi
# Initiate matrix A (2x2), and vector/point u (2x1)
A = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

# Hardcoded (2x2) x (2x1) matrix-vector dot-product function.
def dot_product(A, u):
    dot = np.array([0, 0])
    dot[0] = A[0,0] * u[0] + A[0, 1] * u[1]
    dot[1] = A[1,0] * u[0] + A[1, 1] * u[1]
    return dot

newer_rect = []
for i in range(len(new_rect)):
    newer_rect.append(dot_product(A, new_rect[i]))

for i in range(len(new_rect)):
    plt.plot(new_rect[i][0], new_rect[i][1], 'ro')
    plt.plot(newer_rect[i][0], newer_rect[i][1], 'bo')

fig.show()
