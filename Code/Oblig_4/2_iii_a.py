import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return t


def g(t):
    return t**2


def l():
    # Integral[a, b]{sqrt(f'(t)^2 + g'(t)^2)}
    pass



## For plotting
def c(t):
    x = f(t)
    y = g(t)
    return x, y


t = np.linspace(0, 1, 100)

x, y = c(t)

plt.plot(x, y)
plt.show()