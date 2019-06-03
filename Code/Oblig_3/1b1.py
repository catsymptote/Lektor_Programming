def g(x):
    return 4*x**3 - 10*x - 2


def dg(x):
    return 12*x**2 - 10


def x_intersect(f, df, x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x


print(x_intersect(g, dg, 0))