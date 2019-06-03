# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:22:04 2019

@author: daniell
"""

import numpy as np
import matplotlib.pylab as plt

h = 0.01

t_0 = 0
t_max = 20

# Differensiallikningen y' = F(t, y)

def F(t, y):
    return np.sin(t)*(1-t)*y

def _next_y(t, y):
    return y + F(t, y)*h

def _euler(t_0, t_max, y_0):
    t_lst = [t_0]   # liste med t-verdier, med start t_0
    y_lst = [y_0]   # liste med y-verdier, med start y_0
    lst = np.linspace(t_0, t_max, int((t_max-t_0)/h))  

    for i in range(1,len(lst)): # går gjennom alle punkter
        t_lst.append(t_0 + i*h) # legger vi t-verdiene
        y_lst.append(_next_y(t_0+(i-1)*h, y_lst[i-1])) # beregner neste y-verdi og legger til listen
    return t_lst, y_lst

fig = plt.figure()          
ax1 = fig.add_subplot(111)
ax1.grid(True)

for i in np.arange(1,2,1):
    y_0 = i
    x, y = _euler(t_0, t_max, y_0)
    ax1.plot(x, y)

fig.show()