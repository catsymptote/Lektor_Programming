import numpy as np

def f(x):
    return np.cos(np.e**np.sin(x)/(x**3 + 1)) - 1/2

def df(x):
    return -(np.e**np.sin(x)*(-3*x**2 + (1 + x**3)*np.cos(x))*np.sin(np.e**np.sin(x)/(1 + x**3)))/(1 + x**3)**2

def newtons_method(x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x

x = newtons_method(1)
print("(x, y) : (" + str(x) + ", " + str(round(f(x), 4)) + ")")

##  Med x=-1 får du problemer grunnet at både f og df
##  har nevnere x^3 + 1. Hvis x=-1 får du
##  (-1)^3 + 1 = -1 + 1 = 0. Deling på 0 er ikke definert,
##  og f og df er derfor ikke definert når x=-1.

##  https://www.wolframalpha.com/input/?i=cos(e%5Esin(x)%2F(x%5E3+%2B+1))+-+1%2F2
##  https://www.wolframalpha.com/input/?i=-(e%5Esin(x)+(-3+x%5E2+%2B+(1+%2B+x%5E3)+cos(x))+sin(e%5Esin(x)%2F(1+%2B+x%5E3)))%2F(1+%2B+x%5E3)%5E2&lk=1&assumption=%22ClashPrefs%22+-%3E+%7B%22Math%22%7D
