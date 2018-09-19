from __future__ import division
import numpy as np

#Solving the "normal equations" A.T*A*x = A.T*b with Choleksy factorization 
#Finding x with forward substitution as after factorization matrix is lower triangular 

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

A = np.c_[np.ones((n, 1)), x, x**2]
B = np.dot(A.T, A)

def Cholesky(Matrix):
    n = Matrix.shape[0]
    L = np.zeros((n,n))
    D = np.zeros((n,n))
    B = Matrix
    for k in range(n):
        for j in range(n):
            L[j, k] = B[j, k]/B[k, k]
        D[k, k] = B[k, k]
        B = B - D[k, k]*np.outer(L[:, k], L[:, k].T)   
    return L

L = np.linalg.cholesky(B)
print(L)

R = Cholesky(B)
print(R)
#C = np.dot(R, R.T)
#c = A.T.dot(y(x))

compareX = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(y(x))
#print (compareX)

def ForwardSubstitution(A, b):
    n = len(b)
    x = np.zeros(n)
    if A[n-1, n-1] == 0:
        raise ValueError

    for i in range(n):
        temp = 0
        for j in range(i):
            temp += A[i, j]*x[j]
        x[i] = (b[i] - temp)/A[i, i]
    return x
#print(ForwardSubstitution(R, c))
