# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:37:58 2019

@author: daniell
"""

import matplotlib.pylab as plt

import numpy as np

def f(x):
    y = -2*x**2+1
    return y

g = lambda x: x*np.sin(2*x)

#for i in np.linspace(-2,2,10):
#    print(f(i))
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

x = np.linspace(-2,2,100) # skale i x-retning  intervall [-2,2]


plt.plot(x,f(x),g(x))

fig.show()