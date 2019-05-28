import numpy as np

def f(t):
    return t**2


def integrate(a, b):
    d = 10000
    dt = 1/d
    integral = 0
    for i in range(d-1):
        t = (i*dt)*(b-a) + a
        integral += f(t)*dt*(b-a)
    
    return integral


print(integrate(0, 1))
