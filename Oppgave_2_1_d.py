#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:22:07 2019

@author: catsymptote
"""



# P(x) for single number input R^1
p = [1, 3, -2, 6, 4, -9, 1, 8]
def P(x):
    result = 0
    for i in range(len(p)):
        result += p[i] * x**i
    return result

def P_rec(a, n):
    if n == 1:
        return P(a)
    else:
        return P_rec(a, n-1)

a = 2
n = 3

def P_vector(n_max):
    resulting_vector = [0] * n
    resulting_vector[0] = a
    for i in range(n_max - 1):
        resulting_vector[i+1] = P_rec(a, i)
    return resulting_vector

print(P_vector(n))