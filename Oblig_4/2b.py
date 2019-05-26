import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.cos(t)


def g(t):
    return np.sin(t)


def c(t):
    x = f(t)
    y = g(t)
    return x, y


t = np.linspace(0, 2*np.pi, 100)

x, y = c(t)

plt.plot(x, y)
plt.show()