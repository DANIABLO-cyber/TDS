# Practica 2: PROCESOS AR Y DENSIDAD ESPECTRAL DE POTENCIA
# Autores: Miguel Angel Parrilla Buendía y Guillermo Ruvira Quesada.
# %%
# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz
from scipy.linalg import inv

# Considérese el siguiente proceso AR(2):

# x(n) = -a1 * x(n-1) - a2 * x(n-2) + v(n)

# donde v(n) es un ruido blanco de media nula y varianza sigma_v^2 = 1,
# y los coeficientes son:
a1 = 0
a2 = 0.81
b0 = 1

sigma_v = 1


def autocorrelacion_estimada(x, k_vals, N):
    r_x_estimada = np.zeros(len(k_vals))

    for i in range(len(k_vals)):  # i toma los valores de los índices
        k = k_vals[i]  # Obtener el valor correspondiente de k_vals

        termino1 = x[
            abs(k) :
        ]  # x[n] para n >= |k|. Extrae desde el valor absoluto de k hasta el final

        termino2 = x[
            : N - abs(k)
        ]  # x[n] para n < N - |k|. Extrae desde el principio hasta N - |k|

        r_x_estimada[i] = (1 / N) * np.sum(termino1 * np.conjugate(termino2))
        # Calcular la autocorrelación

    # Misma expresion como convolucion 1/n · x[k] * x[-k]
    r_x_estimada_conv = np.convolve(x, np.conjugate(x[::-1]), "full") / N
    return r_x_estimada, r_x_estimada_conv


def autocorrelacion_sin_sesgo(x, k_vals, N):
    r_x_correlation = np.correlate(x, x, mode="full")

    r_x_sin_sesgo = np.zeros_like(r_x_correlation)
    for i, k in enumerate(k_vals):
        r_x_sin_sesgo[i] = r_x_correlation[i] / (
            N - abs(k)
        )  # escalamos por N - |k| este factor amplifica las fluctuaciones cuando k es

    return r_x_sin_sesgo


# %%
# Ejercicio 1: Obtener y dibujar la autocorrelación r_x(k) y el espectro P_x(w) del proceso AR(2).Deducir la autocorrelación con las ecuaciones de Yule-Walker.

# Definir el número de muestras
N = 32

Matriz_yulee_walker = np.array([[1, 0, a2], [1, 1 + a2, 0], [a2, 0, 1]])
terminos_independientes = np.array([1, 0, 0])

# Calcular los coeficientes de la ecuación de Yule-Walker producto matricial
r0, r1, r2 = np.dot(inv(Matriz_yulee_walker), terminos_independientes)

k = np.arange(-N + 1, N)

# Calcular la autocorrelación recursivamente

rx = np.zeros(2 * N - 1)
rx[N - 1] = r0
rx[N] = r1
rx[N + 1] = r2

# calculamos los valores de la autocorrelacion de manera regresiva k > 2 r(x) = -a1*rx(x-1)-a2*rx(x-2)
for i in range(N - 1, len(rx) - 2):
    rx[i + 2] = -a1 * rx[i + 1] - a2 * rx[i]

# Para muestras negativas aplicamos simetría
for i in range(N - 1):
    rx[i] = rx[2 * N - 2 - i]  # tamaño 2*N-1 por lo que ultimo indice 2*N-2

N = 1024  # tamaño fft

# Calcular la transformada de Fourier de la autocorrelación
P = np.fft.fft(rx, N)
P = np.abs(P) ** 2
# Obtener las frecuencias correspondientes
frequencies = np.fft.fftfreq(N)

# Ordenamos las frecuencias las negativas a la izquierda y positivas a la derecha
frequencies = np.fft.fftshift(frequencies)
P = np.fft.fftshift(P)

# PSD estimada a partir de la expresión
w = np.linspace(-np.pi, np.pi, N)  # vector de frecuencias

denominador = np.abs(1 + a1 * np.exp(-1j * w) + a2 * np.exp(-2j * w)) ** 2
x = len(denominador)
P_estimada = sigma_v**2 / denominador

# estimamos la autocorrelacion como la inversa de la transformada de fourier de P_estimada
rx_estimada = np.fft.ifft(P_estimada, N).real  # cogemos solo la parte real
rx_estimada = np.fft.fftshift(rx_estimada)  # ordenamos frecuencias

k_est = np.arange(-N // 2, N // 2)

# Representación gráfica

# Graficamos el espectro de potencia
plt.figure()
plt.plot(frequencies, P)
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(w)$")
plt.title("Espectro de Potencia")
plt.grid()
plt.show()

# Representamos
plt.figure()
plt.stem(k, rx)
plt.title("Autocorrelación")
plt.xlabel("k")
plt.ylabel("$r_x(k)$")
plt.grid()
plt.show()

# Graficamos la autocorrelación estimada
plt.figure()
plt.stem(k_est[N // 2 - 40 : N // 2 + 40], rx_estimada[N // 2 - 40 : N // 2 + 40])
plt.title("Autocorrelación estimada")
plt.xlabel("k")
plt.ylabel("$r_x(k) estimada$")
plt.grid()
plt.show()

# Graficamos el espectro de potencia
plt.figure()
plt.plot(w, P_estimada)
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(w) estimada$")
plt.title("Espectro de Potencia estimado")
plt.grid()
plt.show()


# %%
# Ejercicio 2: Generar una realización del proceso AR(2) de duración N=32 a partir de 1000 muestras filtradas.
# Conservar solo las últimas N muestras para evitar efectos transitorios.

# generamos ruido blanco gaussiano

muestras = 1000
ruido = np.random.randn(muestras)

# Coeficientes del filtro las dadas
b = [b0]  # Numerador
a = [1, a1, a2]  # Denominador

x = signal.lfilter(b, a, ruido)

# nos quedamos con las últimas 32 muestras
x = x[muestras - 32 :]
n = np.arange(32)

# representamos las últimas 40 muestras
plt.figure()
plt.stem(n, x)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Ultimas 32 muestras de x[n]")
plt.grid()
plt.show()


# %%
# Ejercicio 3: Estimar la autocorrelación utilizando las expresiones con y sin sesgo.
# Comparar gráficamente con la autocorrelación teórica.


N = 32

k_vals = np.arange(-N + 1, N)

# utilizamos la funciones creadas en la primera practica

rx_estimada_con_sesgo, _ = autocorrelacion_estimada(x, k_vals, N)
rx_estimada_sin_sesgo = autocorrelacion_sin_sesgo(x, k_vals, N)

# representamos ambas
plt.figure()
plt.stem(k_vals, rx_estimada_con_sesgo)
plt.xlabel("n")
plt.ylabel("$r_x[k]$")
plt.title("Autocorrelación con sesgo")
plt.grid()
plt.show()

plt.figure()
plt.stem(k_vals, rx_estimada_sin_sesgo)
plt.xlabel("n")
plt.ylabel("$r_x[k]$")
plt.title("Autocorrelación sin sesgo")
plt.grid()
plt.show()

# Representamos
plt.figure()
plt.stem(k, rx)
plt.title("Autocorrelación real")
plt.xlabel("k")
plt.ylabel("$r_x(k)$")
plt.grid()
plt.show()


# %%
# Ejercicio 4: Usar las autocorrelaciones estimadas para calcular las PSDs mediante la transformada de Fourier.
# Comparar con el espectro teórico.

N = 1024

# Calcular la transformada de Fourier de la autocorrelación estimada con sesgo
P_estimada_cs = np.fft.fft(rx_estimada_con_sesgo, N)
P_estimada_cs = np.abs(P) ** 2
# Obtener las frecuencias correspondientes
frequencies_cs = np.fft.fftfreq(N)

# Ordenamos las frecuencias las negativas a la izquierda y positivas a la derecha
frequencies_cs = np.fft.fftshift(frequencies_cs)
P_estimada_cs = np.fft.fftshift(P)

# Calcular la transformada de Fourier de la autocorrelación estimada sin sesgo
P_estimada_ss = np.fft.fft(rx_estimada_con_sesgo, N)
P_estimada_ss = np.abs(P) ** 2
# Obtener las frecuencias correspondientes
frequencies_ss = np.fft.fftfreq(N)

# Ordenamos las frecuencias las negativas a la izquierda y positivas a la derecha
frequencies_ss = np.fft.fftshift(frequencies_ss)
P_estimada_ss = np.fft.fftshift(P)

# representamos
# Graficamos el espectro de potencia
plt.figure()
plt.plot(frequencies_cs, P_estimada_cs)
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(w)$")
plt.title("Espectro de Potencia estimado con sesgo")
plt.grid()
plt.show()

# Graficamos el espectro de potencia
plt.figure()
plt.plot(frequencies_ss, P_estimada_ss)
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(w) estimada$")
plt.title("Espectro de Potencia estimado")
plt.grid()
plt.show()


# %%
# Ejercicio 5: A partir de la autocorrelación sesgada, estimar los parámetros del proceso AR(2)
# resolviendo las ecuaciones de Yule-Walker (matriz Toeplitz). Comentar los resultados.
N = 32
r = rx_estimada_con_sesgo[N - 1 : N + 2] # valores de interés

tpz = toeplitz(r[0:2])

indep = np.array([r[1], r[2]])

parametros = -np.dot(inv(tpz), indep)

print(f"a1 = {parametros[0]} a2 = {parametros[1]}")



# %%
# Ejercicio 6: Con los parámetros estimados del proceso AR(2), obtener la PSD estimada con la expresión y compararla
# con la PSD original y la obtenida con el periodograma.


a1 = parametros[0]
a2 = parametros[1]

N = 1024
# PSD estimada a partir de la expresión
w = np.linspace(-np.pi, np.pi, N)  # vector de frecuencias

denominador = np.abs(1 + a1 * np.exp(-1j * w) + a2 * np.exp(-2j * w)) ** 2

# Graficamos el espectro de potencia
plt.figure()
plt.plot(w, P_estimada)
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(w) estimada$")
plt.title("Espectro de Potencia estimado")
plt.grid()
plt.show()


# %%
# Ejercicio 7: Repetir varias veces la ejecución de los apartados anteriores y comentar los resultados.
# voy a hacer un bucle que nos repita n veces la rx_estimada y vamos a calcular la varianza y la media
iteraciones = 1000
N_rx = 32
muestras = 1000
k_vals = np.arange(-N_rx + 1, N_rx)

a1s = np.zeros(iteraciones)
a2s = np.zeros(iteraciones)

for i in range(0, iteraciones):
    ruido = np.random.randn(muestras)
    # Coeficientes del filtro las dadas
    b = [b0]  # Numerador
    a = [1, a1, a2]  # Denominador
    x = signal.lfilter(b, a, ruido)
    # nos quedamos con las últimas 32 muestras
    x = x[muestras - 32 :]
    rx_estimada_con_sesgo, _ = autocorrelacion_estimada(x, k_vals, N_rx)
    r = rx_estimada_con_sesgo[N_rx - 1 : N_rx + 2] # valores de interés
    tpz = toeplitz(r[0:2])
    indep = np.array([r[1], r[2]])
    parametros = -np.dot(inv(tpz), indep)
    a1s[i] = parametros[0]
    a2s[i] = parametros[1]
    

a1s_media = np.mean(a1s)
a2s_media = np.mean(a2s)

a1s_varianza = np.var(a1s)
a2s_varianza = np.var(a2s)

print(f"\nMedia de a1: {a1s_media}, Varianza de a1: {a1s_varianza}")
print(f"Media de a2: {a2s_media}, Varianza de a2: {a2s_varianza}")

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(a1s, label='a1')
plt.plot(a2s, label='a2')
plt.title('Evolución de los coeficientes a1 y a2 en 100 repeticiones')
plt.xlabel('Iteración')
plt.ylabel('Valor del coeficiente')
plt.legend()
plt.grid(True)
plt.show()

    
    
    
    