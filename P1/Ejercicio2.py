import numpy as np
import matplotlib.pyplot as plt
from scalib import UniformSQ

# Parámetros de la señal sinusoidal
fs = 8000  # Frecuencia de muestreo en Hz
T = 1  # Duración en segundos
f = 50  # Frecuencia de la señal en Hz

# Generar la señal sinusoidal
t = np.arange(0, T, 1/fs)
x_sin = np.sin(2 * np.pi * f * t)

# Parámetros de cuantificación
xRange = (-1, 1)  # Rango de la señal
b = 3  # Bits de cuantificación

# Crear cuantificador UniformSQ de media contrahuella (midrise)
qtz_sin_midrise = UniformSQ(b, xRange, qtype="midrise")

# Cuantificar la señal sinusoidal
xq_sin_midrise = qtz_sin_midrise.quantize(x_sin)

# Representar la señal original y cuantificada durante 1 ciclo (160 muestras)
plt.figure(figsize=(10, 5))
plt.plot(t[:160], x_sin[:160], label="Original", linestyle="dotted", alpha=0.7)
plt.step(t[:160], xq_sin_midrise[:160], label="Cuantizada (Midrise)", where="mid", alpha=0.9)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Cuantificación de Señal Sinusoidal - 1 Ciclo (50 Hz, 8 kHz, 3 bits, Midrise)")
plt.legend()
plt.grid()
plt.show()

# Análisis de niveles de cuantificación
unique_levels = np.unique(xq_sin_midrise)
num_levels = len(unique_levels)
has_zero = 0 in unique_levels

# Resultados del análisis
print(f"Número de niveles de cuantificación: {num_levels}")
print(f"Niveles de cuantificación observados: {unique_levels}")
print(f"¿Existe el nivel 0?: {'Sí' if has_zero else 'No'}")

# Calcular el error de cuantificación
e_q = x_sin - xq_sin_midrise

# Potencia del error de cuantificación
P_e_q = np.mean(e_q ** 2)
P_e_q_dB = 10 * np.log10(P_e_q)

# Potencia teórica del error de cuantificación
delta = (xRange[1] - xRange[0]) / (2 ** b)
P_teorica = (delta ** 2) / 12
P_teorica_dB = 10 * np.log10(P_teorica)

# Mostrar resultados
print(f"Potencia del error de cuantificación: {P_e_q_dB:.2f} dB")
print(f"Potencia teórica del error de cuantificación: {P_teorica_dB:.2f} dB")


# Calcular la FFT y centrar el espectro
X_sin = np.fft.fftshift(np.fft.fft(x_sin))
X_q_sin = np.fft.fftshift(np.fft.fft(xq_sin_midrise))
freqs = np.fft.fftshift(np.fft.fftfreq(len(x_sin), 1/fs))

# Graficar el espectro centrado de la señal original y cuantificada
plt.figure(figsize=(10, 5))
plt.plot(freqs, 20 * np.log10(np.abs(X_sin)), label="Original")
plt.plot(freqs, 20 * np.log10(np.abs(X_q_sin)), label="Cuantizada", linestyle="dashed")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title("Espectro Centrado de la Señal Original y Cuantificada")
plt.legend()
plt.grid()
plt.show()
# Explicación del resultado
print("\nExplicación del resultado del espectro:")
print("- El espectro de la señal original muestra un único pico en 50 Hz, como se espera de una señal sinusoidal pura.")
print("- El espectro de la señal cuantificada conserva ese pico principal, pero también aparecen componentes adicionales (ruido) en frecuencias altas.")
print("- Estas componentes adicionales son armónicos introducidos por el error de cuantificación, y se distribuyen a lo largo del espectro.")
print("- A mayor número de bits de cuantificación, menor será el error y más se parecerá el espectro cuantificado al original.")
