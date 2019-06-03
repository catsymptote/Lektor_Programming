#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:49:49 2019

@author: catsymptote
"""

# Imports
import matplotlib.pyplot as plt

# Function f
def f(x):
    #return x+1
    return x**2 - x + 1

# Function f list generator
def f_list(x_0):
    x_list = []
    x_list.append(x_0)
    for i in range(10):
        x_1 = f(x_0)
        x_list.append(x_1)
        x_0 = x_1
    return x_list

# Function g
def g(x):
    return 0.7*x - 1

# Function g list generator
def g_list(y_0):
    y_list = []
    y_list.append(y_0)
    for i in range(10):
        y_1 = g(y_0)
        y_list.append(y_1)
        y_0 = y_1
    return y_list

# Creating lists of xs and ys
x_list = f_list(0.2)
y_list = g_list(0.8)

# Plot/graphing
plt.plot(x_list, y_list, '-o')
plt.grid()
plt.show()
