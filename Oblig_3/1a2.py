import numpy as np

def f(x):
    return np.log(x**3 + 1) - x**2/2

def df(x):
    return x*(3*x/(x**3 + 1) - 1)

def newtons_method(x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x

x = newtons_method(1)
print("(x, y) : (" + str(x) + ", " + str(f(x)) + ")")

##  https://www.wolframalpha.com/input/?i=d%2Fdx+ln(x%5E3+%2B+1)+-+x%5E2%2F2
