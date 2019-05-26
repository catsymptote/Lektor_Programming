import numpy as np

def e_sq(a, b):
    return 1 - b**2/a**2


def f(a, b, theta):
    return np.sqrt(1 - e_sq(a, b) * np.sin(theta)**2)


def arc_length(a, b):
    d = 10000
    integral = 0
    for i in range(d):
        theta = i/d * np.pi/2
        integral += f(a, b, theta)
    
    return 4*a*integral/d*np.pi/2


print(arc_length(10, 5))
print("48.44224110273838 (W|A)")
