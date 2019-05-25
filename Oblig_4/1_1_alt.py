import numpy as np

def zeta(m, s):
    result = 0
    for n in range(m):
        result += 1/np.power(n+1, s)
    
    return result


"""
x = []
y = []
z = []

k = 0
for i in range(100):
    for j in range(20):
        res = zeta(i, j)
        x = i
        y = j
        z = res
        k += 1
"""

# Plotting
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(2, 50, 50)
y = np.linspace(1, 10, 50)
z = []

X, Y = np.meshgrid(x, y)
print(X)

Z = zeta(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


ax.view_init(60, 35)
fig
