import numpy as np
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


def estimar_pitch_autocorrelacion(e, fs, fmin=50, fmax=500):
    """
    Estima el pitch de una señal usando autocorrelación.

    Parámetros:
    - e: señal de entrada (1D)
    - fs: frecuencia de muestreo
    - fmin, fmax: rango de búsqueda del pitch [Hz]

    Devuelve:
    - f0: pitch estimado en Hz
    - lag_pitch: lag correspondiente
    """
    # * Calcular autocorrelación (sesgada o normalizada)
    autocorr = np.correlate(e, e, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # nos quedamos con la mitad positiva

    # * Limitar búsqueda al rango de lags correspondiente a [fmax, fmin]
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)

    # * Buscar el máximo local en ese rango
    lag_pitch = np.argmax(autocorr[min_lag:max_lag]) + min_lag

    # * Calcular frecuencia fundamental
    f0 = fs / lag_pitch

    return f0, lag_pitch


def coeficientes_AR_YW(x, p):
    """
    Estima los coeficientes del modelo AR(p) y la varianza del ruido blanco
    usando las ecuaciones de Yule-Walker para k>p.

    Parámetros:
    x : array_like
        Señal de entrada (serie temporal)
    p : int
        Orden del modelo AR

    Retorna:
    A : ndarray
        Coeficientes AR incluyendo A[0] = 1
    sigma2 : float
        Varianza estimada del ruido blanco
    """

    N = len(x)

    # Calcular la autocorrelación sesgada de la señal
    r_x_biased = autocorr_biased(x)

    # Construir matriz Toeplitz R con r(0) hasta r(p-1)
    R = toeplitz(r_x_biased[N - 1 : N - 1 + p])

    # Vector del lado derecho: r(1) hasta r(p), negado
    r = -r_x_biased[N : N + p]

    # Resolver el sistema R·a = -r para obtener coeficientes AR
    A = np.linalg.solve(R, r)
    A = np.concatenate(([1.0], A))  # * Importante: A[0] = 1 por convención

    # Calcular la varianza del ruido blanco (sigma^2)
    sigma2 = r_x_biased[N - 1] + np.dot(A[1:], r_x_biased[N : N + p])

    return A, sigma2


def filtro_fir_LS(x, p):
    N = len(x)

    col = x[p - 1 : N - 3]
    fil = x[: p - 2 :]  # O el segmento adecuado según el modelo(va al revés)
    fil = fil[::-1]  # O el segmento adecuado según el modelo(va al revés)

    X = toeplitz(col, fil)
    X_H = X.conj().T  # Transpuesta conjugada

    d = x[p : N - 2]
    Rx = np.dot(X_H, X)  # producto matricial
    Rx_inv = inv(Rx)
    rdx = np.dot(X_H, d)

    A = -1 * np.dot(Rx_inv, rdx)

    # rdx0 = np.dot(d.T, d) # energia señal
    rdx_k = rdx[: p - 2]  # ants p-1
    rdx_k_conj = rdx_k.conj()

    A = np.append(1.0, A)

    Emin = rdx[0] - np.dot(A[: p - 2], rdx_k_conj)

    return A, Emin


def coef_cepstrales(vocal, p):
    L = 1024
    A_estimados, sigma2_estimado = coeficientes_AR_yw(vocal, p)
    PSD_LPC = (
        sigma2_estimado / (np.abs(np.fft.fft(A_estimados, L))) ** 2
    )  # C alculado con los parámetros estimados
    PSD_LPC = PSD_LPC[: int(L / 2)]
    PSD_LPC_db = 10 * np.log10(PSD_LPC)
    c_x_LPC = np.fft.ifft(
        PSD_LPC_db, L
    )  # Cepstrum Aplicamos logartimo para descomponer el espectro en dos sumandos.
    coef = c_x_LPC[1:13]  # Selección de los 12 primeros coeficientes

    return coef


def identificar_vocal(vocal_desconocida, p):
    # Cargar las señales de las vocales conocidas
    a = np.loadtxt("voc_a.asc")
    e = np.loadtxt("voc_e.asc")
    i = np.loadtxt("voc_i.asc")
    o = np.loadtxt("voc_o.asc")
    u = np.loadtxt("voc_u.asc")

    # Diccionario de vocales
    nombres = ["a", "e", "i", "o", "u"]
    vocales = [a, e, i, o, u]

    # Calcular los coeficientes cepstrales de la vocal desconocida
    coef_desconocida = coef_cepstrales(vocal_desconocida, p)

    # Calcular la distancia con cada vocal conocida
    distancias = []  # inicializamos vector
    for vocal_conocida in vocales:
        coef_conocida = coef_cepstrales(vocal_conocida, p)
        distancia = np.linalg.norm(coef_desconocida - coef_conocida)
        distancias.append(
            distancia
        )  # append es unç método que añade un elemento al final de una lista

    # Encontrar el índice de la vocal con menor distancia
    indice_minimo = np.argmin(distancias)

    # Devolver el nombre de la vocal identificada
    return nombres[indice_minimo]


def coeficientes_AR_cov(x, p):
    # Diapositiva 47 TEMA 3

    N = len(x)
    q = 1
    s = max(q, p - 1)

    # La primera fila (fila) se toma de:
    # fila = x[s-p+2 : s] y luego se invierte.
    fila = x[s - p + 2 : s]
    fila = fila[::-1]

    # La primera columna (columna) se toma de:
    # columna = x[s : N-1]
    columna = x[s : N - 1]

    # Construir la matriz Toeplitz
    X = toeplitz(columna, fila)
    X_H = X.conj().T  # Transpuesta conjugada

    # Matriz de covarianza de los regresores:
    R_x = np.dot(X_H, X)
    R_x_inv = inv(R_x)

    # Definir el vector x_s1 = x[s+1 : N]
    x_s1 = x[s + 1 : N]

    # Calcular la correlación cruzada: r_x_cov = X_cov^H * x_s1
    r_x_cov = np.dot(X_H, x_s1)

    # Resolver el sistema para obtener a_cov (coeficientes sin incluir el 1)
    A = -np.dot(R_x_inv, r_x_cov)

    # Concatenar el coeficiente 1 al inicio para obtener el vector total de coeficientes AR

    # Calcular E_min según Spectrum:
    # E_min = ||x_s1||^2 + sum(a_cov * r_x_cov) np dot en este caso calcula la norma cuadratica
    Emin = np.dot(x_s1, x_s1) + np.dot(A, r_x_cov)

    A = np.concatenate(([1.0], A))

    return A, Emin
