from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def zeta(m, s):
    result = 0
    for n in range(m):
        result += 1/np.power(n+1, s)
    
    return result


def zeta_function(delta=0.05):
    x = y = np.arange(1, 8, delta)
    X, Y = np.meshgrid(x, y)
    
    M = len(x)
    S = len(y)
    Z = np.zeros(shape=(M, S))
    for m in range(M):
        for s in range(S):
            Z[m][s] = zeta(m, s)
    
    return X, Y, Z


'''
# Built into axes3d.
def get_test_data(delta=0.05):

    from matplotlib.mlab import  bivariate_normal
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z
'''


# Plotting (3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#x, y, z = axes3d.get_test_data(0.05)
x, y, z = zeta_function(0.1)

ax.plot_wireframe(x, y, z, rstride=5, cstride=5)

plt.show()
