import numpy as np
import matplotlib.pyplot as plt
from scalib import UniformSQ, genDither

# === Parámetros de la señal ===
fs = 8000        # Frecuencia de muestreo en Hz
T = 1            # Duración en segundos
f = 50           # Frecuencia de la senoide en Hz
t = np.arange(0, T, 1/fs)
x_sin = np.sin(2 * np.pi * f * t)  # Señal original

# === Parámetros de cuantificación ===
b = 3
xRange = (-1, 1)
qtz = UniformSQ(b, xRange, qtype="midrise")
delta = (xRange[1] - xRange[0]) / (2 ** b)  # Paso de cuantificación

# === Tipos de dither a usar ===
tipos_dither = ['rectangular', 'triangular', 'gaussian']

# === Bucle por cada tipo de dither ===
for tipo in tipos_dither:
    print(f"\n=== DITHER TIPO: {tipo.upper()} ===")

    # 1. Generar y añadir dither a la señal
    dither = genDither(len(x_sin), tipo, delta)
    x_sin_dith = x_sin + dither

    # 2. Cuantificar señal con dither
    xq_dith = qtz.quantize(x_sin_dith)

    # 3. Representación temporal de un ciclo
    plt.figure(figsize=(10, 4))
    plt.plot(t[:160], x_sin[:160], label="Original", linestyle="dotted", alpha=0.7)
    plt.step(t[:160], xq_dith[:160], label=f"Cuantificada + {tipo}", where="mid", alpha=0.9)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title(f"Señal Original vs Cuantificada con Dither ({tipo}) - 1 Ciclo")
    plt.legend()
    plt.grid()
    plt.show()

    # 4. Potencia experimental del dither
    P_dither_exp = np.mean(dither ** 2)

    # 5. Potencia teórica del dither (según el tipo)
    if tipo == 'rectangular':
        P_dither_teo = (delta ** 2) / 12
    elif tipo == 'triangular':
        P_dither_teo = (delta ** 2) / 6
    elif tipo == 'gaussian':
        P_dither_teo = (delta ** 2) / 4

    # 6. Error de cuantificación (entre señal original y cuantificada)
    e_q_dith = x_sin - xq_dith
    P_error_exp = np.mean(e_q_dith ** 2)
    P_error_teo = (delta ** 2) / 12 + P_dither_teo  # asumiendo independencia

    # 7. Mostrar potencias en dB
    print(f"Potencia del dither (exp): {10*np.log10(P_dither_exp):.2f} dB")
    print(f"Potencia del dither (teo):  {10*np.log10(P_dither_teo):.2f} dB")
    print(f"Potencia del error (exp):  {10*np.log10(P_error_exp):.2f} dB")
    print(f"Potencia del error (teo):  {10*np.log10(P_error_teo):.2f} dB")

    # 8. Espectro FFT centrado
    N = len(x_sin)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    X_orig = np.fft.fftshift(np.fft.fft(x_sin)) / N
    X_q = np.fft.fftshift(np.fft.fft(xq_dith)) / N

    # 9. Representación del espectro
    plt.figure(figsize=(10, 5))
    eps = 1e-10
    plt.plot(freqs, 20 * np.log10(np.maximum(np.abs(X_orig), eps)), label="Original")
    plt.plot(freqs, 20 * np.log10(np.maximum(np.abs(X_q), eps)), label=f"Cuantificada + {tipo}", linestyle="dashed")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.title(f"Espectro FFT Centrado - Dither tipo: {tipo.capitalize()}")
    plt.xlim(-500, 500)
    plt.legend()
    plt.grid()
    plt.show()

    # 10. Comentario rápido
    print("Observación:")
    print("- El dither ayuda a reducir la distorsión armónica (líneas finas en el espectro).")
    print("- A cambio, añade ruido de fondo distribuido a lo largo del espectro.")

