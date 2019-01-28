#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:57:35 2019

@author: catsymptote
Oppgave 2.1, b
"""

p = [1, 3, -2, 6, 4, -9, 1, 8]

def P(x):
    result = 0
    for i in range(len(p)):
        result += p[i] * x**i
    return result

print(P(2))