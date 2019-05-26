import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return t


def g(t):
    return t**2


def c(t):
    x = f(t)
    y = g(t)
    return x, y


t = np.linspace(0, 1, 100)

x, y = c(t)

plt.plot(x, y)
plt.show()