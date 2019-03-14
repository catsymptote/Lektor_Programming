import matplotlib.pylab as plt
import numpy as np

def f(x):
    y = -2 * x**2 + 1
    return y

#for i in np.linspace(-2, 2, 10):
#    print(f(i))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

x = np.linspace(-2, 2, 100 -1)  # from x is element in [-2, 2], with 10 different values.
#y = f(x)    # Simple numpy-way of doing this without a for loop.
#plt.plot(x, y)#, marker='*')

plt.plot(x, f(x))
fig.show()
