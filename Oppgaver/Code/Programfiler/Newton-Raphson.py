# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:57:01 2019

@author: daniell
"""

import numpy as np

def f(x):
    return x**2-2

def fprime(x):
    return 2*x

def _newton(x_n):
    return x_n - f(x_n)/fprime(x_n)

x0 = 2

x = [x0]

for i in range(0,11):
    x.append(_newton(x[i]))
print(x)
    


