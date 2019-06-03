def x_intersect(f, df, x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x
