# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

"""
# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
"""
"""
# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
"""


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def zeta(m, s):
    result = 0
    #print(m)
    for n in range(int(m)):
        result += 1/np.power(n+1, s)
    return result

def f_2(m, s):
    results = []
    print(m)
    print(m[0])
    for i in range(len(s)):
        for j in range(len(m)):
            results.append(zeta(m[j], s[i]))
    
    return results


x = np.linspace(2, 5, 4)
y = np.linspace(2, 5, 4)
#print(x)
#print(y)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


ax.view_init(60, 35)
fig
