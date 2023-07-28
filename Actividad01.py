"""
Seminario de Inteligencia Artificial 2
Actividad 01
"""

import numpy as np

#1
print("Ejercicio 1")
a = np.arange(10,50)
print(a)

#2
print("Ejercicio 2")
a = a[::-1]
print(a)

#3
print("Ejercicio 3")
a = np.arange(0,9)
a = np.reshape(a, (3,3))
print(a)

#4
print("Ejercicio 4")
a = np.array([1,2,4,2,4,0,1,0,0,0,12,4,5,6,7,0])
b = np.where(a != 0)
print(b)

#5
print("Ejercicio 5")
a = np.eye(6)
print(a)

#6
print("Ejercicio 6")
a = np.random.rand(3,3,3)
print(a)

#7
print("Ejercicio 7")
print(np.unravel_index(a.argmin(), a.shape))

#8
print("Ejercicio 8")
a = np.full((10,10),0)
a[0,:] = 1
a[:,9] = 1
a[9,:] = 1
a[:,0] = 1
print(a)

#9
print("Ejercicio 9")
a = np.array([np.arange(0,6)]*5)
print(a)

#10
print("Ejercicio 10")
a = np.random.rand(10)
b = np.random.rand(10)
print(np.array_equal(a,b))

#11
print("Ejercicio 11")
a = np.array([[2,1,-3],[5,-4,1],[1,-1,-4]])
b = np.array([[7],[-19],[4]])
print(np.linalg.solve(a,b))

#12
print("Ejercicio 12")
print(np.version.version)

#13
print("Ejercicio 13")
a = np.random.rand(10)
print(np.sort(a))

#14
print("Ejercicio 14")
a.flags.writeable = False
#a[0] = 0

#15
print("Ejercicio 15")
a = np.random.randint(10, size=(10,2))
x,y = a[:,0], a[:,1]
b = np.sqrt(x**2+y**2)
c = np.arctan2(y,x)
print(b)
print(c)

#16
print("Ejercicio 16")
a = np.mgrid[0:1:.1,0:1:.1]
print(a)

#17
print("Ejercicio 17")
x = np.random.randint(1, 15, size=(4))
y = np.random.randint(15, 30, size=(5))
x = np.reshape(x, (-1,1))
c = 1/(x-y)
print(c)

#18
print("Ejercicio 18")
print(np.unravel_index(np.argmax(c, axis=None), c.shape))

