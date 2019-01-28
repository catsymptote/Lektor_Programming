# Import numpy for its arrays.
import numpy as np

# Initiate matrices A and B with values, as numpy arrays.
A = np.array([[1,2,3], [4,5,6], [7,8,9]])
B = np.array([[4,5,6], [7,8,9], [1,2,3]])

# Function which returns dot product from two input matrices.
def dot_product(A, B):
    # Create initial empty 0-matrix. Size is hardcoded here. Fix!
    dot = np.array([[0,0,0], [0,0,0], [0,0,0]])
    # i and j are loops going through each element of the output matrix.
    for i in range(len(A)):
        for j in range(len(A[0])):
            # k-loop goes through each element of the
            #   vector dot product operation for each output-cell.
            for k in range(len(A[0])):
                # Cumulative addition f each element of the dot product.
                dot[i,j] += A[i,k] * B[k,j]
    return dot

# Store and print dot product.
c = dot_product(A, B)
print(c)