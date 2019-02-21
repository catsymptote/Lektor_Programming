#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:24:57 2019

@author: catsymptote
"""

def f(x):
    return x**3

for i in range(1,10):
    print("f({0}) = {1}".format(i, f(i)))
