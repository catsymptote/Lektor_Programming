from sympy import *
import numpy as np


x = Symbol('x')
f = np.cos(np.e**np.sin(x)/(x**3 + 1)) - 1/2


def f(x):
    return np.cos(np.e**np.sin(x)/(x**3 + 1)) - 1/2


def g(x):
    return x*( (3*x)/(x**3 + 1) -1)


def dg(x):
    return (x**6 + 3*x**4 + 2*x**3 - 6*x + 1) / (x**3 + 1)**2


def x_intersect(f, df, x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x


print(x_intersect(g, dg, 0))