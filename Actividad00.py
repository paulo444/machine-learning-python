"""
Seminario de Inteligencia Artificial 2
Actividad 00
"""

import numpy as np

def matrixPower(a, n):
    result = np.array([[1,1],[1,0]], dtype=object)
    while n > 1:
        if n % 2 == 0:
            a = np.dot(a,a)
            n = int(n/2)
        else:
            result = np.dot(a, result)
            a = np.dot(a,a)
            n = int((n-1)/2)
    return np.dot(a, result)
    
def fib(n):
    if n == 0 or n == 1:
        return n
    
    a = np.array([[1,1],[1,0]], dtype=object)
    
    a = matrixPower(a, n)
    
    return a[0][0]

fibonacci = int(input())
res = fib(fibonacci-3)
print(res)