# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:26:53 2019

@author: daniell
"""

import numpy as np

A = np.array([[1,2],[2,-3]])

B = np.array([[-1,5],[4,2]])

C = np.zeros((np.shape(A)[0],np.shape(B)[1]))

for j in range(0,np.shape(B)[1]):
    for i in range(0,np.shape(A)[0]):
       C[i,j] = sum(A[i,:]*B[:,j])
    
print(C)

print(A.dot(B))

print(np.dot(A,B))