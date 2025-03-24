import numpy as np
from scalib import UniformSQ
import matplotlib.pyplot as plt

fs = 8000 # Establece una frec. de muestreo de 8 kHz
t = np.arange(0, 2, 1/fs) # Vector de tiempos de 2 s de duración
x = np.sin(2 * np.pi * 100 * t) # Genera una señal sinusoidal de 100 Hz, x
xRange = (-1, 1) # Rango de la señal x
b = 3 # Tasa de bits por muestra tras la cuantificación
qtz = UniformSQ(b, xRange) # Crea el cuantificador uniforme
xq = qtz.quantize(x)
plt.figure(figsize=(10, 4))
plt.plot(t[:200], x[:200], label="Señal original", linestyle="dashed")
plt.step(t[:200], xq[:200], label="Señal cuantificada", where="mid")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.show()