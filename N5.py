import numpy as np

def Succesive_OverRelaxation(a,b,x,w,it):
    n = len(a)
    for k in range(it):

        for i in range(0, n):

            d = b[i]

            for j in range(0, n):
                if (i != j):
                    d -= a[i][j] * x[j]

            x[i] = (1 - w) * x[i] + (w/a[i][i])*(d)
    return x

def jacobi(A, b, n):

    x = np.zeros(len(A[0]))

    D = np.diagflat(np.diag(A))
    Dinv = np.linalg.inv(D)
    R = A - D

    for i in range(n):
        x = Dinv @ (b - R @ x)

    return x
def relaxation(a,b,gamma,n):

    x = np.zeros(len(a[0]))


    for i in range(0, n):
        x = x + gamma * (b - (a @ x))

    return x

def Gauss_Seidel(A, b, n ):

    x = np.zeros(len(A))
    L = np.tril(A)
    L_1 = np.linalg.inv(L)
    U = np.triu(A,1)

    for i in range(n):
        x = L_1 @ (b - (U @ x))

    return x

n = 3
# initial solution depending on n(here n=3)
x = [0, 0, 0]
a = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
b = [2, 6, 2]


print("Metoda relaksacyjna (Richardsona)")
print(relaxation(a,b,0.2, 50))

print("Metoda Jacobiego")
print(jacobi(a, b, 25))

print("Metoda Gaussa - Seidla")
print(Gauss_Seidel(a,b,25))

print("Metoda Succesive OverRelaxation")
print(Succesive_OverRelaxation(a, b, x, 0.5,90))



