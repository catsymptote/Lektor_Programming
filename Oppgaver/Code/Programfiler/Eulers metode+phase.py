# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:09:19 2019

@author: daniell
"""

import numpy as np
import matplotlib.pylab as plt
h = 0.001

t_0 = 0
t_max = 10

def F(t,y,yprime):          # y'' = F(t,y,y')
    return  -y

### ----- De tre neste funkjonene trenger dere ikke å skjønne -----

def A(t, y):
    return [y[1], F(t,y[0],y[1])]

def _next_y(t, y):
    return y + np.dot(A(t, y),h)

def _euler(t_0, t_max, y_0):
    t_lst = [t_0]
    z_lst = [y_0]
    y_lst = [y_0[0]]
    yprime_lst = [y_0[1]]
    lst = np.linspace(t_0, t_max, int((t_max-t_0)/h))

    for i in range(1,len(lst)):
        t_lst.append(t_0 + i*h)
        z_lst.append(_next_y(t_0+(i-1)*h, z_lst[i-1]))
        y_lst.append(z_lst[i][0])
        yprime_lst.append(z_lst[i][1])
    return t_lst, y_lst, yprime_lst

#### -----  -----
    

fig = plt.figure()              # definerer to koordinatsystem
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax2 = fig.add_subplot(122)
ax2.grid(True)

y_0 = [1,1]     # startverdier y_0 = [y(0), y'(0)]
x, y, yprime = _euler(t_0, t_max, y_0)

ax1.plot(x, y)
ax2.plot(y,yprime)

fig.tight_layout()
fig.show()