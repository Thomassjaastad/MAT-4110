import numpy as np

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
