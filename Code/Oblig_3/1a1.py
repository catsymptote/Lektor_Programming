def f(x):
    return x**4 - 5*x**2 - 2*x + 3

def df(x):
    return 4*x**3 - 10*x - 2

def newtons_method(x):
    for i in range(100):
        x = x - f(x) / df(x)
    return x

x = newtons_method(0)
print("(x, y) : (" + str(x) + ", " + str(f(x)) + ")")

##  https://www.wolframalpha.com/input/?i=x**4+-+5*x**2+-+2*x+%2B+3
