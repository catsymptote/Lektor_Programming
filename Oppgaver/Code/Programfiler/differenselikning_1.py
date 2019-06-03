# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:19:00 2019

@author: daniell
"""
import numpy as np
import matplotlib.pylab as plt

def f(x):
    return x**2-x+1

def g(x):
    return 0.7*x+1

""" Funksjonen h bruker dere i oppgave 2 """

def h(point):                # point er en liste
    x, y = point    
    x_neste = 1.01*y        # neste x-koordinat
    y_neste = -0.6*x + y    # neste y-koordinat
    return [x_neste, y_neste]

x0 = 0.2
y0 = 0.75
lst = [x0]
lst2 = [y0]


fig = plt.figure()          # koordinatsystem
ax = fig.add_subplot(111)
ax.grid(True)
#
for i in range(1,10):
    lst.append(f(lst[i-1])) # beregner og legger til i listen
    lst2.append(g(lst2[i-1]))
    ax.scatter(lst[i],lst2[i], marker = '*')    

ax.plot(lst,lst2)
lst3 = [[x0,y0]]

#for i in range(1,10):
#    # Hvordan skal vi printe alle punkter?
#    lst3.append(h(lst3[i-1]))    
#    ax.scatter(lst3[i][0],lst3[i][1], marker = '*')
#    ax.plot([lst3[i-1][0],lst3[i][0]],[lst3[i-1][1],lst3[i][1]], color = 'black')


print(lst3)


    