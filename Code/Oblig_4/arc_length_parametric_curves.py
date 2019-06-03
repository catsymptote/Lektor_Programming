import numpy as np

def df(t):
    """ g(t) = t"""
    return 1


def dg(t):
    """f(t) = t^2"""
    return 2*t


def f(t):
    return np.sqrt(df(t)**2 + dg(t)**2)


def arc_length(a, b):
    d = 10000
    dt = 1/d
    integral = 0
    for i in range(d-1):
        t = (i*dt)*(b-a) + a
        integral += f(t)*dt*(b-a)
    
    return integral


print(arc_length(0, 1))
print("1.4789428575445974 (W|A)")
