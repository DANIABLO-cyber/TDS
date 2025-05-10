# Practica 4: Cancelador de ECO acústico
# Autor: Guillermo Ruvira Quesada y Miguel Angel Parrilla Buendia

# %% Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.signal as signal
import scipy.io.wavfile as wav


# %% Primera cuestión: Calculo de la snr entre local y signal pract4/p4_audios/


def snr(x, y):
    """
    Calcula la relación señal a ruido (SNR) entre una señal
    y una distorsión aditiva.

    Parámetros:
    x (numpy.ndarray): Señal 1.
    y (numpy.ndarray): Señal 2.

    Retorna:
    float: El valor de la SNR en decibelios (dB).
    """
    x = x.astype(np.float64)  # Aseguramos que las señales sean de tipo float64
    y = y.astype(np.float64)

    error = y - x  # Calculamos el error entre la señal y la distorsión

    # Calculamos la SNR usando la fórmula, la normalización la incluyo para que sea una función general (notesé que en el caso de distorsión aditiva no es necesaria)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((error) ** 2))

    return snr


# Leemos los archivos de audio
fs, local = wav.read("data/local.wav")  # Señal local
fs, signal_ej = wav.read("data/signal.wav")  # Señal transmitida

# Calculamos la SNR entre la señal local y la señal transmitida
snr_local_signal = snr(local, signal_ej)
print(f"SNR entre local y signal: {snr_local_signal:.2f} dB")

# %% Aplicamos el filtro de cancelación de eco adaptativo NLMS


def nlms_filter(x, d, u, p, graficar=False, limit=None):
    """
    Implementa el filtro adaptativo NLMS (Normalized Least Mean Squares).

    Parámetros:
    x (numpy.ndarray): Señal de entrada (señal transmitida).
    d (numpy.ndarray): Señal deseada (señal local).
    u (float): Tasa de aprendizaje.
    p (int): Orden del filtro.
    graficar (bool): Si es True, representa la evolución de los coeficientes.

    Retorna:
    numpy.ndarray: Salida del filtro adaptativo.
    numpy.ndarray: Coeficientes del filtro adaptativo.
    numpy.ndarray: Error entre la señal deseada y la salida del filtro.
    """

    x = x.astype(np.float64)
    d = d.astype(np.float64)

    if limit is None:
        limit = len(x)  # Si no se especifica un límite, usamos la longitud de la señal

    N = len(x)  # Longitud de la señal

    x = np.concatenate(
        (np.zeros(p - 1), x)
    )  # Añadimos ceros al principio de la señal para evitar problemas de indexación

    w = np.zeros(p).astype(np.float64)  # Inicializamos los coeficientes del filtro
    y = np.zeros(N).astype(np.float64)  # Inicializamos la salida del filtro
    e = np.zeros(N).astype(np.float64)  # Inicializamos el error

    if graficar:
        w_evolution = []  # Almacenamos la evolución de los coeficientes

    for n in range(N):
        x_n = x[n : n + p][::-1]  # Ventana de entrada
        y[n] = w.T @ x_n  # Salida del filtro
        e[n] = d[n] - y[n]  # Error

        if n <= limit:  # Actualizamos los coeficientes solo hasta el límite
            mu = u / ((x_n.T @ x_n) + 1e-9)  # Evitamos división por cero
            w += 2 * mu * e[n] * x_n  # Actualización de los coeficientes

        if graficar:
            w_evolution.append(w.copy())  # Guardamos los coeficientes actuales

    if graficar:
        w_evolution = np.array(w_evolution)
        plt.figure(figsize=(10, 6))
        for i in range(p):
            plt.plot(w_evolution[:, i], label=f"Coeficiente {i + 1}")
        plt.title("Evolución de los coeficientes del filtro")
        plt.xlabel("Iteración")
        plt.ylabel("Valor del coeficiente")
        plt.legend()
        plt.grid()
        plt.show()

        # Representación en subplots individuales
        fig, axes = plt.subplots(p, 1, figsize=(10, 2 * p), sharex=True)
        for i in range(p):
            axes[i].plot(w_evolution[:, i])
            axes[i].set_title(f"Evolución del coeficiente {i + 1}")
            axes[i].set_ylabel("Valor")
            axes[i].grid()
        axes[-1].set_xlabel("Iteración")
        plt.tight_layout()
        plt.show()

    return y, w, e


# %% Iteramos para diferentes valores de mu y p
# Valores de mu y p a probar
u_values = [0.0005 * (2**i) for i in range(9)]  # de 0.0005 a 0.128
p_values = range(2, 8)

best_snr = -np.inf
best_mu = None
best_p = None
best_y = None
best_w = None

# convertimos las señales a float64 para evitar problemas de precisión
# Leemos los archivos de audio
fs, local = wav.read("data/local.wav")  # Señal local
fs, signal_ej = wav.read("data/p4_audios/signal.wav")  # Señal transmitida
fs, remota = wav.read("data/p4_audios/remota.wav")  # Señal remota

remota = remota.astype(np.float64)
local = local.astype(np.float64)
signal_ej = signal_ej.astype(np.float64)


# Iteramos sobre los valores de mu y p
snr_values = np.zeros(
    (len(u_values), len(p_values))
)  # Matriz para almacenar los valores de SNR

for i, p in enumerate(p_values):
    for j, u in enumerate(u_values):
        y, w, e = nlms_filter(remota, signal_ej, u, p)
        current_snr = snr(local, e)  # SNR entre la señal local y el error
        snr_values[j, i] = current_snr  # Guardamos el SNR en la matriz
        # print(f"u: {u}, p: {p}, SNR: {current_snr} dB")

        # Guardamos el mejor caso
        if current_snr > best_snr:
            best_snr = current_snr
            best_mu = u
            best_p = p
            best_w = w
            best_e = e

# Representamos la mejor salida del filtro con la función nmls_filter
print(f"Mejor SNR: {best_snr:.2f} dB con u={best_mu} y p={best_p}")
y, w, e = nlms_filter(remota, signal_ej, best_mu, best_p, graficar=True)

# Representación snr en funcion de p y u

u_mesh, p_mesh = np.meshgrid(p_values, u_values)

fig, ax = plt.subplots(figsize=(10, 6))
norm = Normalize(vmin=np.min(snr_values), vmax=np.max(snr_values))
c = ax.contourf(p_mesh, u_mesh, snr_values, levels=50, cmap="plasma", norm=norm)
fig.colorbar(c, ax=ax, label="SNR (dB)")
ax.set_title("Dependencia del SNR en función de u y p")
ax.set_xlabel("Orden del filtro (p)")
ax.set_ylabel("Tasa de aprendizaje (u)")
plt.show()

# Guardamos la señal con eco cancelado en un archivo WAV
wav.write("signal_canceled_tarea2.wav", fs, e.astype(np.int16))

# %% Tercera cuestión:
# Como el apartado anterior pero limitamos a 2150 muestras momento en el que se detecta voz

u_values = [0.0005 * (2**i) for i in range(9)]  # de 0.0005 a 0.128
p_values = range(2, 8)

# convertimos las señales a float64 para evitar problemas de precisión
# Leemos los archivos de audio
fs, local = wav.read("data/local.wav")  # Señal local
fs, signal_ej = wav.read("data/signal.wav")  # Señal transmitida
fs, remota = wav.read("data/remota.wav")  # Señal remota

remota = remota.astype(np.float64)
local = local.astype(np.float64)
signal_ej = signal_ej.astype(np.float64)


# Iteramos sobre los valores de mu y p
snr_values = np.zeros(
    (len(u_values), len(p_values))
)  # Matriz para almacenar los valores de SNR

for i, p in enumerate(p_values):
    for j, u in enumerate(u_values):
        y, w, e = nlms_filter(remota, signal_ej, u, p, limit=2150)
        current_snr = snr(local, e)  # SNR entre la señal local y el error
        snr_values[j, i] = current_snr  # Guardamos el SNR en la matriz
        # print(f"u: {u}, p: {p}, SNR: {current_snr} dB")

        # Guardamos el mejor caso
        if current_snr > best_snr:
            best_snr = current_snr
            best_mu = u
            best_p = p
            best_w = w
            best_e = e

# Representamos la mejor salida del filtro con la función nmls_filter
print(f"Mejor SNR: {best_snr:.2f} dB con u={best_mu} y p={best_p}")
y, w, e = nlms_filter(remota, signal_ej, best_mu, best_p, graficar=True, limit=2150)

# snr en funcion de p y u

u_mesh, p_mesh = np.meshgrid(p_values, u_values)

fig, ax = plt.subplots(figsize=(10, 6))
norm = Normalize(vmin=np.min(snr_values), vmax=np.max(snr_values))
c = ax.contourf(p_mesh, u_mesh, snr_values, levels=50, cmap="plasma", norm=norm)
fig.colorbar(c, ax=ax, label="SNR (dB)")
ax.set_title("Dependencia del SNR en función de u y p")
ax.set_xlabel("Orden del filtro (p)")
ax.set_ylabel("Tasa de aprendizaje (u)")
plt.show()


# Guardamos la señal con eco cancelado en un archivo WAV
wav.write("signal_canceled_tarea3.wav", fs, e.astype(np.int16))
# Representamos la respuesta en frecuencia del filtro FIR obtenido
w_freq, h_freq = signal.freqz(best_w, worN=8000, fs=fs)

plt.figure(figsize=(10, 6))
plt.plot(w_freq, 20 * np.log10(abs(h_freq)), label="Respuesta en frecuencia")
plt.title("Respuesta en frecuencia del filtro FIR")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()
plt.legend()
plt.show()


# Sobre el porque sale mejor, que es evidente en terminos cualitativos y cuantitativos,
# es por que la estimación del tercer apartado se hace solo con eco (no hay voz), y una vez empieza la voz
# se filtra todo con el filtro estimado cuando no habia voz
