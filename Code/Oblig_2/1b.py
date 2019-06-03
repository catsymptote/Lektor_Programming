# Vector describing/defining the polynomial function.
p = [-2, 1, -2, 1]

# Function of x (scalar) and p (vector)
def P(x):
    y = 0
    for i in range(len(p)):
        y += p[i] * x**i
    return y

print(P(3))