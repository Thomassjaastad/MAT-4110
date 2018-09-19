from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import random
import matplotlib.patches as mpatches
#Solving Ax = b with QR factorization
#First data set
n = 30
start = -2
stop = 2
x = np.linspace(start, stop, n)
eps = 1                              #amount of noise
r = np.random.random((n))*eps

def y(x):
    term1 = x*(np.cos(r + 0.5*x**3))
    term2 = x*(np.sin(r + 0.5*x**3))
    return term1 + term2

#End first data set
#Second data set
def y2(x):
    return 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
#End second data set

#Finding R and c for equation R_1x = c
#Initiating for first data set
A = np.c_[np.ones((n, 1)), x, x**2]
Q, R = np.linalg.qr(A)
b = y(x)
c = Q.T.dot(b)
comparex = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)  #Analytic solution. Can compare with x found by back sub

#Initiating for second data set
b2 = y2(x)
c2 = Q.T.dot(b2)
comparex2 = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b2)

m = 3
#SOLVING Ax = b FOR UPPER TRIANGULAR MATRIX A, BACKWARD SUBSTITUTION
def BackSubstitution(A, b):
    n = len(b)    
    x = np.zeros(n)
    if A[n-1, n-1] == 0:
        raise ValueError

    for i in range(n - 1, -1 , -1):  #i = n, n-1,....,1
        temp = 0
        for j in range(i + 1, n):  
            temp +=  A[i, j]*x[j]    
        x[i] = (b[i] - temp)/A[i, i]            
    return x

xSoldata1 = BackSubstitution(R,c)
xSoldata2 = BackSubstitution(R,c2)

def poly(coeffs, x, n, m):
    print(m)
    fit = np.zeros(n)
    for i in range(n):
        for j in range(m):
            fit[i] += coeffs[j]*x[i]**(j)
    return fit

fitfunc1 = poly(xSoldata1, x, n, m)        
fitfunc2 = poly(xSoldata2, x, n, m)

#Plotting
plt.plot(x, fitfunc1, label='Polyfit of degree %d' %(m-1))
plt.plot(x, y(x), 'o', label='Dataset 1')
plt.legend()
plt.show()

plt.plot(x, fitfunc2, label= 'Polyfit of degree %d' %(m-1))
plt.plot(x, y2(x), 'o', label = 'Dataset 2')
plt.legend()
plt.show()
