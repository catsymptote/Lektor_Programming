# Generell

def recursive_fraction(max_val, i=0):
    if i >= max_val:
        return max_val
    
    return i + (i+1)/recursive_fraction(max_val, i+2)

print(recursive_fraction(6))
# 29/76 = 0.3815789.....
