import matplotlib.pyplot as plt
import numpy as np


def f(t, a):
    return a * np.cos(t)


def g(t, b):
    return b * np.sin(t)


def c(t, a, b):
    x = f(t, a)
    y = g(t, b)
    return x, y


t = np.linspace(0, 2*np.pi, 100)

a = -2
b = -3

x, y = c(t, a, b)

plt.plot(x, y)
plt.show()

# a describes the x-scale. b describes the y-scale.
# The ellipse will have an r in the y-axis of 2*b, from -b to b.
# The ellipse will have an r in the x-axis of 2*a, from -a to a.
# Whether a and/or b are positive or negative does not affect the result.
