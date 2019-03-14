import numpy as np

def f2(x):
    return np.log(x**3 + 1) - x**2/2

def df2(x):
    return x*(3*x/(x**3 + 1) - 1)

def newtons_method(x):
    for i in range(20):
        x = x - f2(x) / df2(x)
    return x

x = newtons_method(0)
print("(x, y) : (" + str(x) + ", " + str(f2(x)) + ")")