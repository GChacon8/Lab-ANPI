from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

def matrizEntrenamiento():
    numFolder = 40
    numImg = 9
    S = []

    for k in range(1, numFolder + 1):
        for i in range(1, numImg + 1):
            direccion = f'training/s{k}/{i}.jpg'
            T1 = io.imread(direccion) / 1.0
            T2 = T1.flatten()
            S.append(T2)

    return S

def obtPromedio(S):
    sum = S[0]

    for i in range(1, 360):
        sum = sum + S[i]

    F = sum / 360
    return F

def crearMatrizA(S, F):
    A = []
    for i in range(0, 360):
        T = S[i] - F
        A.append(T)
    return A

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

def crearMatrizX(U, A):
    X = []
    for i in range(0, 360):
        X.append(U.T * A[i])
    return X

def reconocimientoFacial(X, V, F):

    numImages = 40
    for i in range(1, numImages + 1):
        image = f'compare/p{i}.jpg'
        img = io.imread(image) / 1.0
        f_aux = img.flatten()
        f = V.T * (f_aux - F)
    
        imgI = 1
        min = np.linalg.norm(f - X[0], ord = 1)

        for i in range(1, 360):
            norm = np.linalg.norm(f - X[i], ord = 1)
            if norm < min:
                imgI = i + 1
                min = norm

        print(imgI)
        carpeta = (imgI-1) // 9 + 1
        archivo = imgI - ((carpeta - 1) * 9)
        print(carpeta, archivo)
        direccion = f'training/s{carpeta}/{archivo}.jpg'
        imgSimilar = io.imread(direccion) / 255.0
        mostrarImagenes(img, imgSimilar)
    return 0

def mostrarImagenes(imagenIzquierda, imagenDerecha):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Mostrar la primera imagen a la izquierda
    ax0 = plt.subplot(gs[0])
    ax0.imshow(imagenIzquierda)
    ax0.axis('off')

    # Mostrar la segunda imagen a la derecha
    ax1 = plt.subplot(gs[1])
    ax1.imshow(imagenDerecha)
    ax1.axis('off')

    # Ajustar el diseño y mostrar la figura
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)  # Esperar 1 segundo
    plt.close()

def main():
    S = matrizEntrenamiento()
    F = obtPromedio(S)
    A = crearMatrizA(S, F)
    U, S, V = svdCompact(np.array(A))
    X = crearMatrizX(V, np.array(A))
    reconocimientoFacial(X, V, F)
    return 0

main()
