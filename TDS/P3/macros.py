import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
from scipy.linalg import toeplitz  # comando de generacion matrices Toeplitz
from scipy.linalg import inv  # comando de inversion de matrices


def autocorr_biased(x):
    """
    Calcula la autocorrelación sesgada de una señal x
    Devuelve un arreglo con 2N-1 muestras, centrado en el lag 0
    """
    N = len(x)

    # Autocorrelación de 'x' consigo misma (modo 'full' => incluye todos los lags posibles)
    r_x = np.correlate(x, x, mode="full") / N  # ! Sesgada: se divide por N fijo

    return r_x


def FIR_LS_COV_method_equalizer(x, d, L, i_0, i_f, n_0):
    """
    Calcula los coeficientes del filtro FIR-LS para ecualización mediante el método de covarianza.

    Se minimiza el error entre la señal deseada d(n) (pilotos) y la salida del ecualizador:

        d(n) ≈ y(n) = ∑[k=0]^(L-1) h(k) x(n - n0 - k)

    sobre el intervalo de entrenamiento n = i0, ..., i_f.

    Parámetros:
        x   : Señal recibida.
        d   : Señal piloto (deseada).
        L   : Longitud del filtro FIR (número de coeficientes).
        i_0 : Índice inicial del intervalo de entrenamiento.
        i_f : Índice final (inclusive) del intervalo de entrenamiento.
        n0  : Parámetro de retardo que alinea la señal x con la señal d.

    Retorna:
        h    : Vector de coeficientes del filtro FIR (longitud L).
        E_min: Energía mínima del error en el intervalo de entrenamiento.

    Nota:
        Para cada muestra n en [i0, i_f], se asume que existen los valores necesarios
        en x para obtener el vector de regresores:
        [x(n - n0), x(n - n0 - 1), ..., x(n - n0 - L + 1)].
    """
    # Número de muestras en el intervalo de entrenamiento
    M = i_f - i_0 + 1

    if M >= len(x):
        raise ValueError(
            "El intervalo n∈[i0, if] de entrenamiento es mayor que la longitud de x."
        )
    # Inicializar la matriz de regresores X (dimensión M x L)
    X = np.zeros((M, L), dtype=x.dtype)

    # Construcción de X:
    # Para cada índice n en el intervalo [i0, i_f], la fila corresponde a:
    # [ x(i0), x(i0 - 1), ..., x(n - n0 - L + 1) ]
    for i in range(M):
        n = i_0 + i
        X[i, :] = x[n : n - L]

    # Vector objetivo: señal piloto (deseada) en el intervalo de entrenamiento
    d_train = d[i_0 + n_0 : i_f + n_0 + 1]

    # Matriz de autocorrelación de los regresores y vector de correlación cruzada
    R_x = X.conj().T @ X
    r_dx = X.conj().T @ d_train

    # Resolución del sistema de ecuaciones: R_x * h = r_dx
    h = inv(R_x) @ r_dx

    # Cálculo de la energía mínima del error:
    # E_min = ||d_train||^2 - h^H r_dx
    E_min = np.sum(np.abs(d_train) ** 2) - np.real(np.vdot(h, r_dx))

    return h, E_min


# TODO: asocoewfj
# def FIR_LS_COV_method(x, i_0, i_f, d, n_0):
#         '''
#         asd
#         '''
#     # ?------------------------------------------------------
#     # ? 1. Definiciones básicas
#     # ?------------------------------------------------------
#     N = len(x)  # * Longitud total de la señal
#     if (i_0-i_f) >= N:
#         raise ValueError("ERROR: i_0-i_f >= N")

#     # ?------------------------------------------------------
#     # ? 2. Construcción de la matriz de regresores y del vector objetivo
#     # ?------------------------------------------------------

#     # * Construir la primera columna de X:
#     #   Se toman los valores desde x(p-1) hasta x(N-2) (inclusive).
#     columna = x[i_0 : i_f+1]  #  x(i0), x(i0+1), ..., x(if)  La longitud es if-ip
#     fila = x[i_0:L][::-1]  # x(p-1), x(p-2), ..., x(0)     La longitud es p

#     # * Construir la matriz de regresores X (dimensión (N-p) x p)
#     X = toeplitz(columna, fila)

#     # * Definir el vector de salida (target) d:
#     d = x[p:N]  # d = [x(p), x(p+1), ..., x(N-1)]

#     # ?------------------------------------------------------
#     # ? 3. Resolución de las ecuaciones normales de covarianza
#     # ?    Se resuelve: (X^H X) a_vec = - X^H d, donde a_vec = [a(1), ..., a(p)]
#     # ?------------------------------------------------------
#     X_H = X.conj().T  # * Transpuesta conjugada de X (útil para datos complejos)
#     R_x = X_H @ X  # * Matriz de autocorrelación de los regresores
#     r_x = X_H @ d  # * Vector de correlación cruzada con la salida
#     A_partial = -inv(R_x) @ r_x  # ! Solución para los coeficientes a(1) ... a(p)

#     # ?------------------------------------------------------
#     # ? 4. Cálculo del error cuadrático mínimo E_p
#     # ?    E_p = ∑[n=p]^(N-1) |e(n)|^2, con e(n) = x(n) + ∑[k=1]^(p) a[k] x(n-k)
#     # ?------------------------------------------------------
#     E_p = np.dot(d, d) + np.dot(A_partial, r_x)

#     # ?------------------------------------------------------
#     # ? 5. Construcción del vector de coeficientes del modelo AR
#     # ?    Se define el polinomio:
#     # ?      A(z) = 1 + ∑[k=1..p] a(k) z^{-k}
#     # ?    Por convención, se añade a[0] = 1.
#     # ?------------------------------------------------------
#     a = np.concatenate(([1.0], A_partial))

#     return a, E_p
