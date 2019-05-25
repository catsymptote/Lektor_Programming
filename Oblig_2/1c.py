# For maths and plotting
import matplotlib.pyplot as plt
import numpy as np

# Vector describing/defining the polynomial function.
p = [-2, 1, -2, 1]

# Function of x (scalar) and p (vector)
def P(x):
    y = 0
    for i in range(len(p)):
        y += p[i] * x**i
    return y

def P_vect(u):
    v = [0]*len(u)
    for i in range(len(u)):
        v[i] = P(u[i])
    return v

u = np.linspace(-10, 10, 101)
y_values = P_vect(u)
x_values = np.linspace(0, len(y_values) -1, len(y_values))

print(x_values)
print(y_values)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

plt.plot(x_values, y_values)
fig.show()

input("Press enter to exit!")  # Too keep the window open.