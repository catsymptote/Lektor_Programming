# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:25:56 2019

@author: paul
"""

import matplotlib.pyplot as plt

# Input matrix and vector
A = [[1, 2], [3, 4]]
u = [2, 1]

# Hard coded dot product (and transpose) function for (2x2) x (2x1) = (2x1).
def dot_product(A, u):
    v = [0]*len(u)
    
    v[0] = A[0][0] * u[0] + A[0][1] * u[1]
    v[1] = A[1][0] * u[0] + A[1][1] * u[1]
    
    return v

# Call dot_product and print out result.
point = dot_product(A, u)
print(point)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
plt.plot(point[0], point[1], 'ro')
fig.show()