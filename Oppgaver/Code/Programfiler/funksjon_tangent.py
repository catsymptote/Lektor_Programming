# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:49:38 2019

@author: daniell
"""

import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-2*np.pi,2*np.pi,100) # Definierer intervallet

h = 0.001                       # h-verdi

def P(t):                       # Definisjon av funksjonen
    return t**2*np.sin(t)  

def _Pprime(t):                 # Den deriverte i punktet x = t
    return (P(t+h)-P(t))/h

def _point(t):                  # Denne funksjon returnerer punktet på grafen med x-koordinat x = t 
    return [t,P(t)]

def _tangent(t, point):         # Input x = t, og punktet der tangenten skal plottes
    x0 = point[0]               # plukker ut x-koordinat for punkt
    y0 = point[1]               # plukker ut y-koordinat for punkt
    return _Pprime(x0)*(t-x0)+y0 # Output y-koordinat på tangenten som korresponderer til x = t

x_in = float(input('x = '))     # Spør etter x-koordinat for punkt der tangenten skal plottes

# skriver ut punktet og den deriverte i punktet:
print("({0}); P'({1})={2}".format(_point(x_in), _point(x_in)[0], _Pprime(x_in)))

# definierer koordinatsystem:

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

ax.plot(x,P(x))  # Plotter grafen til P(x)
ax.scatter(x_in, P(x_in) , marker = '.', s = 100, c = 'green') # Plotter punktet 
ax.plot(x, _tangent(x, _point(x_in))) # Plitter tangenten

fig.show()
