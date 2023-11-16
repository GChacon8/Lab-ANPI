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
        s1 = np.sqrt(y3)
        order = np.argsort(s1)[::-1]
        s1 = s1[order]
        V2 = V1[:, order]
        Vr = V2[:, :rA]
        Sr = np.diag(s1[:rA])
        Ur = (1/(s1[:rA]).T) * np.dot(A, Vr)
    else:
        M1 = np.dot(A, A.T)
        y1, U1 = np.linalg.eig(M1)
        const = m * max(y1) * np.finfo(float).eps
        y2 = (y1 > const)
        rA = np.sum(y2)  # rango de la matriz
        y3 = y1 * y2
        s1 = np.sqrt(y3)
        order = np.argsort(s1)[::-1]
        s1 = s1[order]
        U2 = U1[:, order]
        Ur = U2[:, :rA]
        Sr = np.diag(s1[:rA])
        Vr = (1 /(s1[:rA]).T) * np.dot(A.T, Ur) 

    return Ur, Sr, Vr

'''
def pregunta1():

    A = np.array([[1, 2],
                          [3, 4],
                          [5, 6],
                          [7, 8]])

    U, S, V = np.linalg.svd(A, full_matrices=False)
    print("Matriz U: ", U, end= "\n")
    print("Matriz S: ", S, end= "\n")
    print("Matriz V: ", U, end= "\n")
      

    Ur, Sr, Vr = svdCompact(A)
    print("Matriz Ur: ", Ur, end= "\n")
    print("Matriz Sr: ", Sr, end= "\n")
    print("Matriz Vr: ", Ur, end= "\n")
''' 

def pregunta1():
    k_values = [5, 6, 7, 8, 9, 10, 11, 12]

    tiempos_funcion1 = np.zeros(len(k_values))
    tiempos_funcion2 = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        A = matrizAleatoria(k)

        start_time = time.time()
        Ur, Sr, Vr = np.linalg.svd(A, full_matrices=False)
        print(Ur, Sr, Vr)
        tiempos_funcion1[i] = time.time() - start_time

        start_time = time.time()
        Ur, Sr, Vr = svdCompact(A)
        print(Ur, Sr, Vr)
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
