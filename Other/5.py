#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:18:47 2019

@author: catsymptote

Oppgave:
    Skriv et program som plotter en funksjon.
    Programmet skal ta input fra tastatur et x-verdi,
    og plotte tangenten til grafen funksjonen i (x, f(x)).
"""


# Import matplotlib for graphs and numpy for maths.
import matplotlib.pylab as plt
import numpy as np


# The function itself.
def f(x):
    y = -2 * x**2 + 1
    #y = np.sin(x)
    #y = np.cos(x)
    #y = np.exp(x)
    return y


# Function to find the slope of f(x) at x_1
h = 0.00001
def slope(x_1):
    y = f(x_1 + h) - f(x_1)
    a = y/h
    return a


# Function to find value of the tangent line.
def tangent(a, b, x_1, x):
    y = a*x - a*x_1 + b
    return y


# Make figure and stuff.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

# Get input x_1 (or not).
x_1 = float(input("Input value: "))
#x_1 = 1

# Get a and b of the linear tangent function.
a = slope(x_1)
b = f(x_1)

# Change [-x, x] to fit x_1
minmax = 2
if abs(x_1) * 2 > minmax:
    minmax = abs(x_1) * 2

# Generate linspace at size.
x = np.linspace(-minmax, minmax, 100 -1)

# Plot and show functions
plt.plot(x, f(x), x, tangent(a, b, x_1, x))
fig.show()
