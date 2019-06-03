import math

def continued_fraction(nums, i=0):
    if i >= len(nums) -1:
        return 1/nums[i]
    
    return nums[i] + 1/continued_fraction(nums, i+1)


# Integers
nums = [0,1,2,3,4,5,6,7,8,9,10]
ints = continued_fraction(nums)

# sqrt(2)
nums = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
sqrt2 = continued_fraction(nums)

# sqrt(3)
nums = [1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1]
sqrt3 = continued_fraction(nums)

# e
nums = [2,1,2,1,1,4,1,1,6,1,1,8]
euler = continued_fraction(nums)

# pi
nums = [3,7,15,1,292,1,1,1,2,1,3,1]
pi = continued_fraction(nums)

# phi
nums = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
phi = continued_fraction(nums)
golden_ratio = 1.61803398874989484820
# https://www.mathsisfun.com/numbers/golden-ratio.html


# For printing/table making purposes.
def printer(a, b, c, d, e):
    print(str(a) + "\t\t" + str(b) + "\t" + str(c) + "\t" + str(d) + "\t" + str(e))


printer("Descr", "Answer\t\t", "Estimate\t", "Absolute error\t", "Relative error (%)")
print("-------------------------------------------------------------------------------------------------------------")
printer("sqrt(2)",  math.sqrt(2),   sqrt2,  abs(sqrt2 - math.sqrt(2)),  abs(1 - sqrt2 / math.sqrt(2)))
printer("sqrt(3)",  math.sqrt(3),   sqrt3,  abs(sqrt3 - math.sqrt(3)),  abs(1 - sqrt3 / math.sqrt(3)))
printer("euler",    math.e,         euler,  abs(euler - math.e),        abs(1 - euler / math.e))
printer("pi",       math.pi,        pi,     abs(pi    - math.pi),       abs(1 - pi    / math.pi))
printer("phi",      golden_ratio,   phi,    abs(phi   - golden_ratio),  abs(1 - phi   / golden_ratio))
