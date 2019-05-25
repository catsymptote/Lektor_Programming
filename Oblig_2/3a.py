# Initial values
n = 5           # List size
lst = [0]*n     # Make empty list
x = 0.5
lmda = 0.5

# Function L
def L(x, lmda):
    return lmda * x*(1-x)

# Use L "recursively"
def L_rec(x, lmda, rec_level):
    for i in range(rec_level):
        x = L(x, lmda)
    return x

# Make list
for i in range(n):
    lst[i] = L_rec(x, lmda, i)

# Print list
for i in range(n):
    print(lst[i])


# Virker litt som at med lambda < 1.5
#   blir L^(n)(x) bare mindre når
#   n -> infinity (eller 4),
#   mens med lambda > 1.5 blir L større først,
#   så mindre når n -> infinity (eller 4).