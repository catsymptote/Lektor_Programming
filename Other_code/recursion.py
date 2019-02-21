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

def f(x):
    return x**3

lst = [2]

for i in range(4):
    lst.append(f(lst[len(lst) -1]))
print(lst)
