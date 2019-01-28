import numpy as np

A = np.array([[1,2,3], [4,5,6]])
#B = np.array([[5, 2, 6], [1, 4, 3]])
B = np.array([[5,2], [1,4], [3,4]])
C = np.copy(B) # C = B is not deep copy
B[0,0] = 4
print(C)
#(np.matmul(A, B))

#A = np.array([[1, 2, 3], [6, 7, 9], [-1, -2, 3]])
def scalar_matrix_mult(scalar, A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] *= scalar
    return A
#print(scalar_matrix_mult(2, A))
