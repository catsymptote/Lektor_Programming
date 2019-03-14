def f1(x):
    return x**4 - 5*x**2 - 2*x + 3

def df1(x):
    return 4*x**3 - 10*x - 2

def newtons_method(x):
    for i in range(20):
        x = x - f1(x) / df1(x)
    return x

x = newtons_method(0)
print("(x, y) : (" + str(x) + ", " + str(f1(x)) + ")")