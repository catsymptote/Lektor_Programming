#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:24:57 2019

@author: catsymptote
"""
"""
lst = []

def f(x):
    print(x)
    if len(lst) < 10:
        x = x**3
        lst.append(x)
        return f(x)

f(2)
"""

import numpy as np

def f(x):
    return np.sin(x)

lst = [1.07]

for i in range(100):
    lst.append(f(lst[len(lst) -1] +i))
print(lst)
