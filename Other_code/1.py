import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B_add = np.array([[5, 2, 6], [1, 4, 3]])
B_mul = np.array([[3, 1], [5, 4], [2, 4]])

def matrix_addition(A, B):
    # Matrix shape test
    if(not(A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1])):
        return None
    
    # Get shape and initiate new return matrix
    A_x = A.shape[0]
    A_y = A.shape[1]
    C = np.empty([A_x, A_y])
    
    # Addition
    for i in range(A_x):
        for j in range(A_y):
            C[i][j] = A[i][j] + B[i][j]
    
    # "return A+B" also works, but is Python af
    return C


def vector_scalar_produt(u, v):
    scalar = 0
    for i in range(u.shape[0]):
        scalar += u[i] * v[i]


def matrix_scalar_product(A, B):
    # Matrix shape test
    if(not(A.shape[1] == B.shape[0])):
        return None
    
    # Get shape and initiate new return matrix
    x = A.shape[0]
    y = B.shape[1]
    C = np.empty([x, y])

    for i in range(x):
        for j in range(y):
            # Extract relevant vectors and use vector_scalar_product
            pass
    
    # Cheaty Python AF
    return np.dot(A, B)

C_add = matrix_addition(A, B_add)
C_mul = matrix_scalar_product(A, B_mul)
print(C_add)
print(C_mul)
