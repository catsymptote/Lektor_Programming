#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:15:24 2019

@author: catsymptote
"""

""" Skriv et program som tar som input et startverdi
-1 <= x_0 <= 1.
La f(x) = 1-x^2.
Beregn listen [x_0, x_1, ..., x_n] hvor x_(n+1) = f(x_n).
Plot deretter punktene (x_0, x_1), (x_1, x_2), (x_2, x_3), ...,
i et koordinatsystem.
"""

import numpy as np
import matplotlib.pyplot as plt

# Point resolution/list length, and initial value.
n_max = 20
lst = [-0.7] # -1 <= x <= 1


# Function f.
def f(x):
    return 1 - x**2


# Create x's (x_0, ..., x_n).
for i in range(n_max):
    lst.append(f(lst[len(lst) -1]))


# Plotting.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

# Create the points.
for j in range(n_max):
    plt.scatter(lst[j], lst[j+1])
    #print("({0}, {1})".format(lst[j], lst[j+1]))

fig.show()
