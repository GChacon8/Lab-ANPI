import numpy as np
import matplotlib.pyplot as plt
import time

def matrizAleatoria(k):
    m = 2 ** k
    n = 2 ** (k - 1)
    A = np.random.rand(m, n) * 10000 - 5000
    return A

import numpy as np

def svdCompact(A):
    m, n = A.shape

    if m > n:
        M1 = np.dot(A.T, A)
        y1, V1 = np.linalg.eig(M1)
        const = n * max(y1) * np.finfo(float).eps
        y2 = (y1 > const)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        V2 = V1[:, order]
        Vr = V2[:, :rA]
        Sr = np.diag(np.sqrt(y3[order][:rA]))
        Ur = np.dot(A, Vr) / np.sqrt(y3[order][:rA])
    else:
        M1 = np.dot(A, A.T)
        U1, D = np.linalg.eig(M1)
        y1 = np.diag(D)
        const = m * max(y1) * np.finfo(float).eps
        y2 = (y1 > const)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        order = np.argsort(np.sqrt(y3))[::-1]
        U2 = U1[:, order]
        Ur = U2[:, :rA]
        Sr = np.diag(np.sqrt(y3[order][:rA]))
        Vr = np.dot(A.T, Ur) / np.sqrt(y3[order][:rA])

    return Ur, Sr, Vr



def pregunta1():
    k_values = [5, 6, 7, 8, 9, 10, 11, 12]

    tiempos_funcion1 = np.zeros(len(k_values))
    tiempos_funcion2 = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        A = matrizAleatoria(k)

        start_time = time.time()
        Ur, Sr, Vr = np.linalg.svd(A, full_matrices=False)
        tiempos_funcion1[i] = time.time() - start_time

        start_time = time.time()
        Ur, Sr, Vr = svdCompact(A)
        tiempos_funcion2[i] = time.time() - start_time

    plt.plot(k_values, tiempos_funcion1, '-o', label='SVD Python')
    plt.plot(k_values, tiempos_funcion2, '-o', label='SVD Compact')
    plt.xlabel('Valor de k')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Tiempo de ejecución de funciones en función de k')
    plt.legend()
    plt.grid(True)
    plt.show()

pregunta1() 
