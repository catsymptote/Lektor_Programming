#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:48:15 2019

@author: catsymptote
"""

# Imports
import matplotlib.pyplot as plt

# Function descriotion
def f(x, y, a, b):
    x_next = a*y
    y_next = b*x + y
    return [x_next, y_next]

# Starting values
x = 0.2
y = 0.3
a = 1.01
b = -0.6

# Creating function list
P = []
P.append([x, y])
for i in range(10):
    P_next = f(x, y, a, b)
    x = P_next[0]
    y = P_next[1]
    P.append(P_next)

# Plot/graphing
plt.plot(P, '-o')
plt.grid()
plt.show()
