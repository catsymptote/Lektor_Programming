# Vector describing/defining the polynomial function.
p = [-2, 1, -2, 1]


# Function of x (scalar) and p (vector)
def P(x, n):
    y = 0
    for i in range(len(p)):
        y += p[i] * x**i
    if n == 0:
        return x
    elif n == 1:
        return y
    else:
        return P(y, n-1)


def P_rec(x, n):
    result = [0]*n
    for i in range(n):
        result[i] = P(x, i)
    return result


a = 3   # Input number
n = 5   # Vector size/factorial "loops".
print(P_rec(a, n))