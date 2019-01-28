#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:52:23 2019

@author: catsymptote
"""

# Vector describing/defining the polynomial function.
p = [-2, 1, -2, 1]


# Function of x (scalar) and p (vector)
def P(x, n):
    y = 0
    for i in range(len(p)):
        y += p[i] * x**i
    if n == 0:
        return x
    elif n == 1:
        return y
    else:
        return P(y, n-1)


def P_rec(x, n):
    result = [0]*n
    for i in range(n):
        result[i] = P(x, i)
    return result


def P_rec_vect(u, n):
    v = [0]*len(u)
    for i in range(len(u)):
        v[i] = P_rec(u[i], n)
    return v

u = [0, 1, 2, 3, 4]
n = 5   # Vector size/factorial "loops".
print(P_rec_vect(u, n))