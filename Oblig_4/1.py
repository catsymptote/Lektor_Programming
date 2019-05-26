import numpy as np
import matplotlib.pyplot as plt
import math


def zeta(m, s):
    """Zeta function for a singluar m and s."""
    result = 0
    for n in range(int(m)):
        result += 1/np.power(float(n+1), s)
    
    return result


def zeta_function_m(s=4):
    """Iterative Zeta function for a range of m's."""
    start = 2
    stop = 40
    
    x = np.linspace(start, stop, stop - start + 1)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = zeta(x[i], s)
    
    return x, y


def zeta_function_s(m):
    """Iterative Zeta function for a range of s's."""
    start = -5
    stop = 5
    
    x = np.linspace(start, stop, stop - start + 1)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = zeta(m, x[i])
    
    return x, y


#fig = plt.figure()
x, y = zeta_function_s(2)
#print(x)
plt.plot(x, y)
plt.show()

# zeta_function_m(s)
#   m controls smoothness of curve.
#   s controls bend.
#   s = 0 results in a straight line, from about (0,0) to (m,m)
#   As s goes towards -inf, the curve will bend to the bottom right.
#   As s goes towards +inf, the curve will bend to the top left.

# zeta_function_s(m)
#   s range (stop - start) controls the smoothness of the curve.
#   m < 2 results in a straight horizontal line at 1.
#   As m->+inf, the curve will bend to the bottom left.
#   This will also happen if the s range becomes larger.


### (ii)
print("\n\nEstimation, zeta(2) = pi^2/6")
estimate = zeta(1000, 2)
print(estimate)

actual_value = math.pi**2/6
print(actual_value)

print("Absolutt error: " + str(abs(estimate - actual_value)))


### (iii)
print("\n\nEstimation, zeta(4) = pi^4/90")
estimate = zeta(1000, 4)
print(estimate)

actual_value = math.pi**4/90
print(actual_value)

print("Absolutt error: " + str(abs(estimate - actual_value)))
