#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:51:21 2019

@author: catsymptote
"""

# Function for nx1 vectors, where n != 1
def transpose_n1(M):
    """ M = [0, 0] --> [[0], [0]] """
    x = len(M)
    N = []
    for i in range(x):
        N.append([M[i]])
    return N


# Function for 1xn vectors, where n != 1
def transpose_1n(M):
    """ M = [[0], [0]] --> [0, 0]"""
    N = [0]*len(M)
    for i in range(len(M)):
        N[i] = M[i][0]
    return N


# Function for nxn matrices, where n != 1
def transpose_nn(M):
    N = []
    for i in range(len(M[0])):
        tmp = [0] * len(M)
        for j in range(len(M)):
            #N[i][j] = M[j][i]
            tmp[j] = M[j][i]
        N.append(tmp)
    return N


# Main function
def transpose(M):
    if type(M[0]) != list:
        return transpose_n1(M)
    elif len(M[0]) < 2:
        return transpose_1n(M)
    else:
        return transpose_nn(M)


#print([0, 1, 2])
#print(transpose_n1([0, 1, 2]))
#print(transpose_1n([[0], [1], [2]]))

#print(transpose([[0, 1, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
#print(transpose([3, 2, 1]))
#print(transpose([[2], [1], [0]]))