# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:34:51 2019

@author: daniell
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate

num_points = 20
x, dx = np.linspace(-1, np.pi, num_points, retstep = True)

""" --- Funksjoner --- """

def f(x):
    return np.sin(x**3-x**2)+2*x-1

def _soeyle(f,x,dx):
    return f(x)*dx

def _trapes(f,x,dx):
    if x>=np.pi: 
        return 0
    else:
        trekant = (f(x+dx)-f(x))*dx/2
        return trekant + _soeyle(f,x,dx)

def _integration(f,x,dx):
    int_tot_soeyle = 0
    int_tot_trapes = 0
    for i in x:
        int_tot_soeyle = int_tot_soeyle + _soeyle(f,i,dx)
        int_tot_trapes = int_tot_trapes + _trapes(f,i,dx) 
    return int_tot_soeyle, int_tot_trapes

def _scipy_int(f,x,dx):
    return integrate.quad(f,x[0],x[num_points-1])    

def _integrate(f,x,dx):
    """ Beregning av integraler """
    s, t = _integration(f,x,dx) # Søyler og trapeser
    q = _scipy_int(f,x,dx)      # SciPy integrasjon
    return s, t, q

def _plot(f,x,dx):
    """ Plottning av kurve og søyler """
    fig = plt.figure()          
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)  # koordinatsystem
    ax1.grid(True)                        
    ax2 = fig.add_subplot(133)  # koordinatsystem 2
    ax2.grid(True)

    y = num_points*[0]

    x_plot = np.linspace(-1, np.pi, 100*num_points)
    ax0.plot(x_plot,f(x_plot))
    ax1.plot(x_plot,f(x_plot))  
    ax2.plot(x_plot,f(x_plot))

    for i in x:
        if i < np.pi:
            ax1.scatter(x, y, marker = '*')   
            ax1.scatter(i, f(i), marker = '*')
            ax2.scatter(x, y, marker = '*')   
            ax2.scatter(i, f(i), marker = '*')
            ax1.plot([i, i, i+dx, i+dx], [0, f(i), f(i), 0], color = 'green') 
            ax2.plot([i, i, i+dx],[0, f(i), f(i+dx)], color = 'green') 
        else: 
                ax1.scatter(np.pi, f(np.pi), marker = '*')
                ax1.plot([np.pi, np.pi], [0,f(np.pi)], color = 'green')
                ax2.scatter(np.pi, f(np.pi), marker = '*')
                ax2.plot([np.pi, np.pi], [0,f(np.pi)], color = 'green')
                break

    # Koordinataksler   
    left0, right0 = ax0.get_xlim()
    low0, high0 = ax0.get_ylim()
    left1, right1 = ax1.get_xlim()
    low1, high1 = ax1.get_ylim()    
    left2, right2 = ax2.get_xlim()
    low2, high2 = ax2.get_ylim()
    ax0.plot([left0, right0],[0, 0], color = 'black', linewidth = 2)   
    ax0.plot([0,0], [low0, high0], color = 'black', linewidth = 2)
    ax1.plot([left1, right1],[0, 0], color = 'black', linewidth = 2)   
    ax1.plot([0,0], [low1, high1], color = 'black', linewidth = 2)
    ax2.plot([left2, right2],[0, 0], color = 'black', linewidth = 2)   
    ax2.plot([0,0], [low2, high2], color = 'black', linewidth = 2)

    fig.tight_layout()
    fig.show()
    return True

""" --- main --- """

s, t, q = _integrate(f,x,dx)

print("Søyle = {0}; Trapes = {1};\nAbsolutt differense = {2}".format(s, t, np.abs(t-s)))
print("Integral = {0};\nAbsolutt differense med søyle = {1};\
          \nAbsolutt differense med trapes = {2}".format(q, np.abs(q[0]-s), np.abs(q[0]-t)))

_plot(f,x,dx)