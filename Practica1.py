# Description: Practica 1 de la asignatura de Tratamiento digital de Señales
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Ejercicio 2
"""
Generar N = 1000 muestras de un ruido blanco Gaussiano de media nula y varianza unidad mediante
el comando np.random.randn(N) (MatLab: randn(N)).
"""
N = 1000
ruido = np.random.randn(N)

# Graficar y poner titulos y etiquetas
plt.figure()
plt.plot(ruido)
plt.title("Ruido Blanco Gaussiano")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.show()
# Histograma de las muestras
plt.figure()
plt.hist(ruido, bins=50)
plt.title("Histograma de Ruido Blanco Gaussiano")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.show()

# %%
# Ejercicio 3
"""
Estimar la autocorrelacion rx(k) (k = −(N − 1), . . . , N − 1) de acuerdo con la expresion (1) dada ante-
riormente, y tambien haciendo uso de los comandos convolve (ecuacion (2)) y correlate de numpy
(MatLab: conv y xcorr). Explicar los resultados obtenidos comparando graficamente con la funcion
de autocorrelacion verdadera rx(k) = δ (k).
"""
N = 1000
x = np.random.randn(N)  # Señal x
# Autocorrelacion estimada
k_vals = np.arange(-N + 1, N)  # Valores de k


def autocorrelacion_estimada(x, k_vals):
    r_x_estimada = np.zeros(len(k_vals))

    for i in range(len(k_vals)):  # i toma los valores de los índices
        k = k_vals[i]  # Obtener el valor correspondiente de k_vals
        termino1 = x[abs(k) :]  # x[n] para n >= |k|
        termino2 = x[: N - abs(k)]  # x[n] para n < N - |k|
        r_x_estimada[i] = (1 / N) * np.sum(
            termino1 * np.conjugate(termino2)
        )  # Calcular la autocorrelación
    # Misma expresion como convolucion 1/n · x[k] * x[-k]
    r_x_estimada_conv = np.convolve(x, np.conjugate(x[::-1]), "same") / N
    return r_x_estimada, r_x_estimada_conv


r_x_estimada, r_x_estimada_conv = autocorrelacion_estimada(x, k_vals)

# represenantamos la autocorrelacion estimada
plt.figure()
plt.plot(k_vals, r_x_estimada)
plt.title("Autocorrelación estimada")
plt.xlabel("k")
plt.ylabel("Autocorrelación")
plt.show()


# %%
# Ejercicio 4
"""
Obtener ahora la estima sin sesgo de la autocorrelación usando el comando correlate (MatLab:
xcorr) de numpy. Comparar gráficamente esta estima con la obtenida mediante el estimador sesgado
de los apartados anteriores."""

x = np.random.randn(N)  # Señal x
k_vals = np.arange(-N + 1, N)  # Valores de k


def autocorrelacion_sin_sesgo(x, k_vals):
    r_x_correlation = np.correlate(x, x, mode="full")

    r_x_sin_sesgo = np.zeros_like(r_x_correlation)
    for i, k in enumerate(k_vals):
        r_x_sin_sesgo[i] = r_x_correlation[i] / (N - abs(k))

    return r_x_sin_sesgo


r_x_sin_sesgo = autocorrelacion_sin_sesgo(x, k_vals)

# represenantamos la autocorrelacion estimada color rojo
plt.figure()
plt.plot(k_vals, r_x_sin_sesgo, "r", label="Estimador sesgado")
plt.title("Autocorrelación calculada")
plt.xlabel("k")
plt.ylabel("Autocorrelación")
plt.show()

# por que conforme el lag (k) es mas grande lo que sucede es que tenemos menos parejas de muestras por lo tanto incrementa la varianza que se refleja en la variabilidad de los bordes.
# %%
# Ejercicio 5
"""
Obtener las estimaciones de la autocorrelacion con sesgo y sin sesgo para el mismo ruido anterior
pero añadiendo una media unitaria. ¿Como deberia ser la autocorrelacion original en este caso?
Explicar los resultados obtenidos. 
"""

N = 1000
x = np.random.randn(N) + 1  # Señal x
# Autocorrelacion estimada
k_vals = np.arange(-N + 1, N)  # Valores de k
r_x_estimada, r_x_estimada_conv = autocorrelacion_estimada(x, k_vals)
r_x_sin_sesgo = autocorrelacion_sin_sesgo(x, k_vals)

plt.figure()
plt.plot(k_vals, r_x_estimada)
plt.title("Autocorrelación estimada con media unitaria")
plt.xlabel("k")
plt.ylabel("Autocorrelación estimada con sesgo")
plt.show()

plt.figure()
plt.plot(k_vals, r_x_sin_sesgo, "r", label="Estimador sesgado")
plt.title("Autocorrelación estimada con media unitaria")
plt.xlabel("k")
plt.ylabel("Autocorrelación estimada sin sesgo")
plt.show()

# %%
# Ejercicio 6
"""
Aplicar los estimadores de media y varianza muestral sobre 100 estimas de la autocorrelacion, cada
una de ellas obtenida a partir de una secuencia distinta de ruido blanco de media unidad. ¿Que
sucede si aumentas el numero de estimas a 1000, 10000, etc? ¿A que tiende la media? ¿Que i_ndica
la varianza? Se compararan los resultados obtenidos al usar los estimadores sesgado y no-sesgado.
"""
N = 1000

numero_estimas = 10000
suma_estimaciones = np.zeros(2 * N - 1)
suma_estimaciones_sin_sesgo = np.zeros(2 * N - 1)
producto_estimaciones = np.zeros(2 * N - 1)
producto_estimaciones_sin_sesgo = np.zeros(2 * N - 1)

for i in range(numero_estimas):
    x = np.random.randn(N)
    r_x_estimada, _ = autocorrelacion_estimada(x, k_vals)
    r_x_sin_sesgo = autocorrelacion_sin_sesgo(x, k_vals)
    suma_estimaciones += r_x_estimada
    suma_estimaciones_sin_sesgo += r_x_sin_sesgo
    producto_estimaciones += r_x_estimada * r_x_estimada
    producto_estimaciones_sin_sesgo += r_x_sin_sesgo * r_x_sin_sesgo

media_estimada = suma_estimaciones / numero_estimas
media_estimada_sin_sesgo = suma_estimaciones_sin_sesgo / numero_estimas

varianza_estimada = producto_estimaciones / numero_estimas - media_estimada**2
varianza_estimada_sin_sesgo = (
    producto_estimaciones_sin_sesgo / numero_estimas - media_estimada_sin_sesgo**2
)

plt.figure()
plt.plot(k_vals, media_estimada)
plt.title(f"Media estimada Estimaciones={numero_estimas}")
plt.xlabel("k")
plt.ylabel("Media")
plt.show()

plt.figure()
plt.plot(k_vals, varianza_estimada)
plt.title(f"Varianza estimada Estimaciones={numero_estimas}")
plt.xlabel("k")
plt.ylabel("Varianza")
plt.show()

plt.figure()
plt.plot(k_vals, media_estimada_sin_sesgo)
plt.title(f"Media estimada sin sesgo Estimaciones={numero_estimas}")
plt.xlabel("k")
plt.ylabel("Media sin sesgo")
plt.show()

plt.figure()
plt.plot(k_vals, varianza_estimada_sin_sesgo)
plt.title(f"Varianza estimada sin sesgo Estimaciones={numero_estimas}")
plt.xlabel("k")
plt.ylabel("Varianza sin sesgo")
plt.show()


# %%
