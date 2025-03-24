import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scalib import UniformSQ

# Generación de señal sinusoidal
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

# Crear cuantificadores UniformSQ
qtz_sin_midrise = UniformSQ(b, xRange, qtype="midrise")
qtz_sin_midtread = UniformSQ(b, xRange, qtype="midtread")

# Cuantificar la señal sinusoidal
xq_sin_midrise = qtz_sin_midrise.quantize(x_sin)
xq_sin_midtread = qtz_sin_midtread.quantize(x_sin)

def compute_snr(original, quantized):
    noise = original - quantized
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
    return snr


snr_sin_midrise = compute_snr(x_sin, xq_sin_midrise)
snr_sin_midtread = compute_snr(x_sin, xq_sin_midtread)

# Graficar la señal original y cuantificada
plt.figure(figsize=(10, 5))
plt.plot(t[:200], x_sin[:200], label="Original", linestyle="dotted")
plt.step(t[:200], xq_sin_midrise[:200], label="Cuantizada (Midrise)", where="mid")
plt.step(t[:200], xq_sin_midtread[:200], label="Cuantizada (Midtread)", where="mid")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Cuantificación de Señal Sinusoidal (50 Hz, 8 kHz, 3 bits)")
plt.legend()
plt.grid()
plt.show()
#%%
# Cargar señal de voz
fs_wav, x_wav = wav.read("altura.wav")
x_wav = x_wav / np.max(np.abs(x_wav))  # Normalización

# Crear cuantificadores para la señal de voz
qtz_wav_midrise = UniformSQ(b, (-1, 1), qtype="midrise")
qtz_wav_midtread = UniformSQ(b, (-1, 1), qtype="midtread")

# Cuantificar la señal de voz
xq_wav_midrise = qtz_wav_midrise.quantize(x_wav)
xq_wav_midtread = qtz_wav_midtread.quantize(x_wav)

snr_wav_midrise = compute_snr(x_wav, xq_wav_midrise)
snr_wav_midtread = compute_snr(x_wav, xq_wav_midtread)


# Graficar la señal original y cuantificada
plt.figure(figsize=(10, 5))
plt.plot(np.arange(2000) / fs_wav, x_wav[:2000], label="Original", linestyle="dotted")
plt.plot(np.arange(2000) / fs_wav, xq_wav_midrise[:2000], label="Cuantizada (Midrise)")
plt.plot(np.arange(2000) / fs_wav, xq_wav_midtread[:2000], label="Cuantizada (Midtread)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Cuantificación de Señal de Voz (3 bits)")
plt.legend()
plt.grid()
plt.show()
#%%
# Cargar imagen de Lena
img = plt.imread("lena.png")
if img.ndim == 3:
    img = np.mean(img, axis=2)  # Convertir a escala de grises si es necesario
img = img / np.max(img)  # Normalización

# Crear cuantificadores para la imagen
qtz_img_midrise = UniformSQ(b, (0, 1), qtype="midrise")
qtz_img_midtread = UniformSQ(b, (0, 1), qtype="midtread")

# Cuantificar la imagen
xq_img_midrise = qtz_img_midrise.quantize(img)
xq_img_midtread = qtz_img_midtread.quantize(img)

snr_img_midrise = compute_snr(img, xq_img_midrise)
snr_img_midtread = compute_snr(img, xq_img_midtread)


# Mostrar imágenes cuantificadas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(xq_img_midrise, cmap='gray')
plt.title('Imagen cuantificada (Midrise)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(xq_img_midtread, cmap='gray')
plt.title('Imagen cuantificada (Midtread)')
plt.axis('off')
plt.show()

#MIDTREAD (MEDIA HUELLA): Incluye un nivel en el cero, lo que puede hacer que ciertas 
#zonas de baja intensidad se representen con más precisión.

#MIDRISE (Media CONTRHUELLA): Tiende a representar valores con un pequeño desplazamiento 
#respecto al rango original, lo que puede afectar la luminosidad general.


#%%

# Mostrar imágenes cuantificadas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(xq_img_midrise, cmap='gray')
plt.title('Imagen cuantificada (Midrise)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(xq_img_midtread, cmap='gray')
plt.title('Imagen cuantificada (Midtread)')
plt.axis('off')
plt.show()

# Mostrar resultados de SNR
print("Resultados de SNR:")
print(f"SNR Señal Senoidal - Midrise: {snr_sin_midrise:.2f} dB")
print(f"SNR Señal Senoidal - Midtread: {snr_sin_midtread:.2f} dB")
print(f"SNR Señal de Voz - Midrise: {snr_wav_midrise:.2f} dB")
print(f"SNR Señal de Voz - Midtread: {snr_wav_midtread:.2f} dB")
print(f"SNR Imagen - Midrise: {snr_img_midrise:.2f} dB")
print(f"SNR Imagen - Midtread: {snr_img_midtread:.2f} dB")