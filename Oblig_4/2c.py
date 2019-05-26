import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return t**2


def g(t, A):
    return A * np.sin(2*np.pi*t) * np.cos(np.pi*t)


def c(t, A):
    x = f(t)
    y = g(t, A)
    return x, y


t = np.linspace(-1, 1, 100)

for A in range(5):
    x, y = c(t, A+1)
    plt.plot(x, y)

plt.show()