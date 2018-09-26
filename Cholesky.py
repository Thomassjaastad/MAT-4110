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
b = y(x)
b2 = y2(x)

def Cholesky(Matrix):
    #Matrix must be symmetric 
    n = Matrix.shape[0]  
    L = np.zeros((n,n))
    D = np.zeros((n,n))
    new_A = Matrix
    for k in range(n):
        for j in range(n):
            L[j, k] = new_A[j, k]/new_A[k, k]
        D[k, k] = new_A[k, k]
        new_A = new_A - D[k, k]*np.outer(L[:, k], L[:, k].T)   
    return L, D

L = Cholesky(B)[0]
D = np.sqrt(Cholesky(B)[1])

R = np.dot(L, D)   #Lower triangular matrix, solve Ry = A.Tb, where y = R.Tx

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
 
y_twiddle = ForwardSubstitution(R, np.dot(A.T, b))
y_twiddle2 = ForwardSubstitution(R, np.dot(A.T, b2))

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

x = BackSubstitution(R.T, y_twiddle)
x2 = BackSubstitution(R.T, y_twiddle2)

#Analytic solutions
compareX = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
compareX2 = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b2)

#Printing Solution values
print('-----------------------------------------------------')
print('Cholesky factorization for first data set y(x), solving two equations:')
print('1. Ry_twiddle = A.Tb => y_twiddle = ', y_twiddle)
print('2. R.Tx = y_twiddle => x = ', x)
print('Analytic solution x =      ', compareX)
print('-----------------------------------------------------')
print('-----------------------------------------------------')
print('Cholesky factorization for second data set y2(x), solving two equations:')
print('1. Ry_twiddle2 = A.Tb2 => y_twiddle2 = ', y_twiddle2)
print('2. R.Tx2 = y_twiddle2 => x2 = ', x2)
print('Analytic solution x2 =        ', compareX2)
print('-----------------------------------------------------')

def K(A, x):
    term1 = np.linalg.norm(np.linalg.inv(A), np.inf)
    term2 = np.linalg.norm(A, np.inf)   
    return np.dot(term1, term2)

print ('K(B) = ', K(B, x))
