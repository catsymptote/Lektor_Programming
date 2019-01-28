#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:02:15 2019

@author: catsymptote
Oppgave 2.1, c
"""

import matplotlib.pylab as plt
import numpy as np

# p-vector is the P(x)-function base vector ("p_i")
# u-vector is the input vector ("x").
p = [1, 3, -2, 6, 4, -9, 1, 8]
u = [-4, 1, 5, 9, -1, 8, 1, 6]
#u = [1, 2, 3, 4, 5, 6, 7, 8]

# P(x) for single number input R^1
def P(x):
    result = 0
    for i in range(len(p)):
        result += p[i] * x**i
    return result

# P(u) for multi number input R^n
def P_vector(v):
    res_vect = [0] * len(v)
    for j in range(len(v)):
        res_vect[j] = P(v[j])
    return res_vect


# Make figure and stuff.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

# Plot and show functions
plt.plot(P_vector(u))
fig.show()