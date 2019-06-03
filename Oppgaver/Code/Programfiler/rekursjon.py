# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:24:11 2019

@author: daniell
"""

import numpy as np

def f(x):
    return np.sin(x)

lst = [1.07]

for i in range(1,100):
    lst.append(f(lst[i-1]+i)) 
print(lst)

# print("f({0}) = {1}".format(i, f(i)))