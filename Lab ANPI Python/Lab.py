import numpy as np

def svdCompact(A):
    m, n = A.shape
    if m > n:
        M1 = np.dot(A.T, A)
        D, V1 = np.linalg.eig(M1)
        y1 = np.diag(D)
        const = n * max(y1) * np.finfo(float).eps
        y2 = (y1 > const)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        s1 = np.sqrt(y3[order])
        V2 = V1[:, order]
        Vr = V2[:, :rA]
        Sr = np.diag(s1[:rA])
        Ur = (1 / s1[:rA]) * np.dot(A, Vr)
    else:
        M1 = np.dot(A, A.T)
        D, U1 = np.linalg.eig(M1)
        y1 = np.diag(D)
        const = m * max(y1) * np.finfo(float).eps
        y2 = (y1 > const)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        s1 = np.sqrt(y3[order])
        U2 = U1[:, order]
        Ur = U2[:, :rA]
        Sr = np.diag(s1[:rA])
        Vr = (1 / s1[:rA]) * np.dot(A.T, Ur)
    return Ur, Sr, Vr

matriz = np.array([[2, 1, 2],
                  [2, -2, 2],
                  [-2, -1, -2],
                  [2, 0, 2]])

Ur, Sr, Vr = svdCompact(matriz)