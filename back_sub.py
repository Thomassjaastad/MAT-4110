from __future__ import division
import numpy as np
import matplotlib.pyplot as ply

#SOLVING Ax = b FOR UPPER TRIANGULAR MATRIX A, BACKWARD SUBSTITUTION
def BackSubstitution(A, b):
    n = len(b)    
    x = np.zeros(n)
    if A[n-1, n-1] == 0:
        raise ValueError

    for i in range(n - 1, -1 , -1):                    #i = n, n-1,....,1
        temp = 0
        for j in range(i + 1, n):  
            temp +=  A[i, j]*x[j]    
        x[i] = (b[i] - temp)/A[i, i]            
    return x

A = np.matrix([[1, 0, 1],[0, 1, 1],[0, 0, -2]])
b = np.array([2, 1, 1])

print (BackSubstitution(A,b))
