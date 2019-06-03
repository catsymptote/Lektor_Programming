# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:14:04 2019

@author: daniell
"""

import numpy as np
import matplotlib.pylab as plt

t0 = 0.1                    # startverdi

r = 4                     # r-verdi

def _poly(x, r):            # funksjonen
    return r*x*(1-x)

def _line(x):               # linjen
    return x

t = np.linspace(0,1,100)    # intervall

fig = plt.figure()          # koordinatsystem
ax = fig.add_subplot(111)
ax.grid(True)

ax.plot(t,_poly(t, r))      # parabel
ax.plot(t,_line(t))         # linje

x = [t0]                    # initierer listen med startverdien

for i in range(1,200):
    x.append(_poly(x[i-1],r))                                   # beregner neste punkt og adderer til listen
    ax.plot([x[i-1],x[i-1]], [x[i-1],x[i]], color = 'green')    # linje fra y = x til graf
    ax.scatter(x[i-1],x[i], marker = '*')                       # plotter et punkt
    ax.plot([x[i-1],x[i]],[x[i],x[i]], color = 'green')         # linje fra punkt til y = x
    ax.scatter(x[i],x[i], marker = '*')                         # plotter et punkt    
    
print(x)  # skriver ut listen
plt.show()