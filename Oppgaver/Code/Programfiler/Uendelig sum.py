# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:41:23 2019

@author: daniell
"""

import numpy as np

def f(i):
    return 1/(1+i**2)

def _sum(lst):
    s = 0
    for i in range(0,len(lst)):
        s = s + lst[i]   # viktig programmeringsteknisk skrivemåte
    return s

# Definer en liste med tall

a = []

for i in range(1,100):
    a.append(f(i))

# Summer tallene i denne liste

#print(a)

print(_sum(a))