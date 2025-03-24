import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image

# Definición de la función para el cuantificador de media contrahuella (Mid-Rise)
def midrise_quantizer(x, bits, x_min=-1, x_max=1):
    levels = 2 ** bits  # Número de niveles de cuantización
    q_step = (x_max - x_min) / levels  # Paso de cuantización
    x_q = np.floor(x / q_step + 0.5) * q_step  # Cuantización por redondeo
    return np.clip(x_q, x_min, x_max)  # Se limita al rango permitido

# Definición de la función para el cuantificador de media huella (Mid-Tread)
def midtread_quantizer(x, bits, x_min=-1, x_max=1):
    levels = 2 ** bits  # Número de niveles de cuantización
    q_step = (x_max - x_min) / (levels - 1)  # Paso de cuantización para media huella
    x_q = np.round(x / q_step) * q_step  # Cuantización con redondeo directo
    return np.clip(x_q, x_min, x_max)  # Se limita al rango permitido

# Función para calcular la relación señal a ruido de cuantización (SNR)
def calculate_snr(original, quantized):
    noise = original - quantized  # Se calcula el error de cuantización
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))  # Fórmula de SNR
    return snr

# 1. Generar señal senoidal
fs = 8000  # Frecuencia de muestreo
T = 1  # Duración en segundos
f = 50  # Frecuencia de la señal senoidal

t = np.arange(0, T, 1/fs)  # Vector de tiempos
x_sin = np.sin(2 * np.pi * f * t)  # Generación de la señal senoidal

# Cuantificación de la señal senoidal con ambos cuantificadores
x_sin_midrise = midrise_quantizer(x_sin, 3)
x_sin_midtread = midtread_quantizer(x_sin, 3)

# Cálculo de la SNR para la señal senoidal
snr_sin_midrise = calculate_snr(x_sin, x_sin_midrise)
snr_sin_midtread = calculate_snr(x_sin, x_sin_midtread)

# 2. Cuantificación de señal de voz
x_voice, fs_voice = sf.read('altura.wav')  # Cargar el archivo de audio

# Cuantificar la señal de voz con ambos métodos
x_voice_midrise = midrise_quantizer(x_voice, 3, x_voice.min(), x_voice.max())
x_voice_midtread = midtread_quantizer(x_voice, 3, x_voice.min(), x_voice.max())

# Calcular SNR para la señal de voz
snr_voice_midrise = calculate_snr(x_voice, x_voice_midrise)
snr_voice_midtread = calculate_snr(x_voice, x_voice_midtread)

# Guardar los archivos de audio cuantificados
sf.write('altura_midrise.wav', x_voice_midrise, fs_voice)
sf.write('altura_midtread.wav', x_voice_midtread, fs_voice)

# 3. Cuantificación de imagen
img = np.array(Image.open('lena.png'), dtype=np.float32) / 255.0  # Cargar imagen en color y normalizar

# Cuantificación de la imagen con ambos métodos
img_midrise = midrise_quantizer(img, 3, 0, 1)
img_midtread = midtread_quantizer(img, 3, 0, 1)

# Calcular SNR para la imagen
snr_img_midrise = calculate_snr(img, img_midrise)
snr_img_midtread = calculate_snr(img, img_midtread)

# Mostrar imágenes original y cuantificadas en color
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(img_midrise)
plt.title('Mid-Rise')

plt.subplot(1, 3, 3)
plt.imshow(img_midtread)
plt.title('Mid-Tread')
plt.show()

# Resultados de SNR para cada señal y cuantificador
snr_results = {
    "SNR Sin Mid-Rise": snr_sin_midrise,
    "SNR Sin Mid-Tread": snr_sin_midtread,
    "SNR Voz Mid-Rise": snr_voice_midrise,
    "SNR Voz Mid-Tread": snr_voice_midtread,
    "SNR Imagen Mid-Rise": snr_img_midrise,
    "SNR Imagen Mid-Tread": snr_img_midtread,
}

# Imprimir resultados de SNR
for key, value in snr_results.items():
    print(f"{key}: {value:.2f} dB")
