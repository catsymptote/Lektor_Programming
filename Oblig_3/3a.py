a = 1
b = -1
c = 1
d = 0
e = -0.1
f = 0.1

def F(point):
    x = point[0]
    y = point[1]
    
    xn = a*x + b*y
    yn = c*x + d*y
    
    return [xn, yn]

def F2(point):
    x = point[0]
    y = point[1]
    
    xn = a*x + b*y
    yn = c*x + d*y + e*x**2 + f*y**2
    
    return [xn, yn]


X = [1]
Y = [1]


for i in range(100):
    point = F2([X[i], Y[i]])
    #points.append(F(points[i]))
    X.append(point[0])
    Y.append(point[1])


# Plotting
import matplotlib.pyplot as plt

plt.grid()
plt.plot(X, Y)
plt.show()

#input("Press enter to exit!")  # Too keep the window open.

# Resultat på (d): Blir spiral. Vet ikke hvorfor.
