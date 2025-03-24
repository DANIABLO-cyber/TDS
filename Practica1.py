# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from skimage import io, color
from scalib import UniformSQ
from scalib import signalRange
from scalib import snr
from scalib import genDither
import sounddevice as sd

np.set_printoptions(threshold=np.inf)  # Muestra todo el array

# %% Ejercicio 1:
"""
Construya dos cuantificadores uniformes, uno de media contrahuella y otro de media huella. En
ambos casos la tasa de bits por muestra resultante debe ser de 3 bits. Use ambos cuantificadores
para cuantificar 3 señales:
"""
b = 3

# %%  1.1: Una señal sinusoidal de 50 Hz y 1 s de duración muestreada a 8 kHz.
# ----------------------------------------------------------------------------
fs = 8000
f = 50
t = np.linspace(0, 1, fs)  # 1 segundo
x = np.sin(2 * np.pi * f * t)

xRange = (-1, 1)

qtz_ch = UniformSQ(b, xRange, qtype="midrise")  # cuatificador de media contrahuella
qtz_h = UniformSQ(b, xRange, qtype="midtread")  # cuantificador de media huella

xq_ch = qtz_ch.quantize(x)
xq_h = qtz_h.quantize(x)

# representar seno cuatizado con media contrahuella y media huella
plt.figure()
plt.subplot(121)
plt.plot(xq_ch[:480])
plt.title("Cuantizacion con media \n contrahuella")
plt.subplot(122)
plt.plot(xq_h[:480])
plt.title("Cuantizacion con media huella")
plt.show()

# calculamos las SNR  y las imprimimos funcion snr()
snr_ch = snr(x, xq_ch)
snr_h = snr(x, xq_h)
print(f"SNR con media contrahuella (sin): {snr_ch} dB")
print(f"SNR con media huella (sin): {snr_h} dB")


# %% 1.2: El audio datos/altura.wav cuatizar usando el mismo metodo
# ----------------------------------------------------------------------------
fs, s = wf.read("datos/altura.wav")

# imprimir rango de valores de la señal con la funcion signalRange en scalib.py
wav_range = signalRange(s)

print(range)

qtz_ch = UniformSQ(b, wav_range, qtype="midrise")  # cuatificador de media contrahuella
qtz_h = UniformSQ(b, wav_range, qtype="midtread")  # cuantificador de media huella

sq_ch = qtz_ch.quantize(s)
sq_h = qtz_h.quantize(s)


# Representar ambas cuantizaciones para 0.5 segundos
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(s[fs : fs + fs // 2], label="Original", alpha=0.7)
plt.plot(
    sq_ch[fs : fs + fs // 2], label="Cuantización media contrahuella", linestyle="--"
)
plt.title("Cuantización con media contrahuella (0.5 segundos)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(s[fs : fs + fs // 2], label="Original", alpha=0.7)
plt.plot(sq_h[fs : fs + fs // 2], label="Cuantización media huella", linestyle="--")
plt.title("Cuantización con media huella (0.5 segundos)")
plt.legend()
plt.grid()


plt.show()
# reproducimos sq_h y sq_ch las señales cuantizadas con scipy.io.wavfile.write

print("Reproduciendo señal cuantizada con media contrahuella...")
sd.play(sq_ch, fs)
sd.wait()
sd.stop()

print("Reproduciendo señal cuantizada con media huella...")
sd.play(
    sq_h, fs
)  # como la media huella tiene nivel de cuantizacion 0 para algunos valores de la señal se corta
sd.wait()
sd.stop()

# calculamos las SNR  y las imprimimos funcion snr()
snr_ch = snr(s, sq_ch)
snr_h = snr(s, sq_h)
print(f"SNR con media contrahuella (wav): {snr_ch} dB")
print(f"SNR con media huella (wav): {snr_h} dB")


# %% 1.3: Leer el archivos datos/lena.png y cuantizarlo usando el mismo metodo
# -----------------------------------------------------------------------------
img = io.imread("datos/lena.png")
img_range = signalRange(img)

qtz_ch = UniformSQ(b, img_range, qtype="midrise")  # cuatificador de media contrahuella
qtz_h = UniformSQ(b, img_range, qtype="midtread")  # cuantificador de media huella

imgq_ch = qtz_ch.quantize(img)  # cuantizado con media contrahuella
imgq_h = qtz_h.quantize(img)  # cuantizado con media huella

# representar las dos imagenes cuantizadas
plt.figure()
plt.subplot(121)
plt.imshow(imgq_ch)
plt.title("Cuantizacion con media \n contrahuella")
plt.subplot(122)
plt.imshow(imgq_h)
plt.title("Cuantizacion con media huella")
plt.show()

# calculamos las SNR  y las imprimimos funcion snr()
snr_ch = snr(img, imgq_ch)
snr_h = snr(img, imgq_h)

print(f"SNR con media contrahuella (img): {snr_ch} dB")
print(f"SNR con media huella (img): {snr_h} dB")


# %% Ejercicio 2:
"""
Genere una señal sinusoidal de 50 Hz y 1 s de duración muestreada a 8 kHz. Cuantifí-
quela usando un cuantificador uniforme de media contrahuella con una tasa de 3 bits por
muestra.
"""
b = 3
fs = 8000
f = 50
t = np.linspace(0, 1, fs)  # 1 segundo
x = np.sin(2 * np.pi * f * t)  # señal senoidal

xRange = (-1, 1)

qtz_ch = UniformSQ(b, xRange, qtype="midrise")  # cuantificador de media contrahuella
xq_ch = qtz_ch.quantize(x)  # cuantizamos la señal

"""Represente la señal original y la señal cuantificada durante 1 ciclo (8000/50 = 160 mues-
tras). ¿Cuántos niveles de cuantificación se aprecian en la imagen? ¿Hay algún nivel de
valor 0? ¿Es este el resultado esperado?"""

plt.figure()
plt.subplot(121)
plt.plot(x[:160])
plt.title("Senal original")
plt.subplot(122)
plt.plot(xq_ch[:160])
plt.title("Senal cuantificada")
plt.show()

# calculamos los niveles de cuatización (será 8 evidentemente 2^b = 8) siendo b el numero de bits

niveles = np.unique(xq_ch)

print(f"Tenemos {len(niveles)} nivel de cuantización")


""" A continuación, calcule la potencia del error (varianza) de cuantificación (en decibelios) y compárela
con la potencia teórica de dicho error."""

# calculamos la potencia del error de cuantificacion
Perror_exp = np.var(x - xq_ch)
Perror_dB_exp = 10 * np.log10(Perror_exp)

# calculamos el tamaño del cuantizador y el valor teorico de la probablidad de error
q = (xRange[1] - xRange[0]) / (2**b)  # Tamaño del cuantificador uniforme
P_e_teorico_ch = q**2 / 12
P_e_teorico_dB = 10 * np.log10(P_e_teorico_ch)

print(f"Potencia del error de cuantificacion experimental: {Perror_dB_exp} dB")
print(f"Potencia del error de cuantificacion: {P_e_teorico_dB}")


"""Por último, calcule y represente el espectro de la señal original y el de la señal cuantificada.
Explique el resultado."""

# FFT de la señal original y la cuantificada
X = np.fft.fft(x)
Xq = np.fft.fft(xq_ch)

# Frecuencias correspondientes
freq_X = np.fft.fftfreq(len(x), d=1 / fs)
freq_Xq = np.fft.fftfreq(len(xq_ch), d=1 / fs)

# Aplicar fftshift para centrar el espectro
X = np.fft.fftshift(X)
Xq = np.fft.fftshift(Xq)
freq_X = np.fft.fftshift(freq_X)
freq_Xq = np.fft.fftshift(freq_Xq)

# Representación del espectro completo centrado
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(freq_X, 10 * np.log10(np.abs(X) ** 2))  # evitar log(0)
plt.title("Espectro señal original (completo)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freq_Xq, 10 * np.log10(np.abs(Xq) ** 2))
plt.title("Espectro señal cuantificada (completo)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()

plt.show()

# %% Ejercicio 3:
"""Ejercicio 2 pero añadiendo dithering"""
# Añadimos los 3 tipos de dithering.
# Generamos la señal:
fs = 8000
f = 50
t = np.linspace(0, 1, fs)  # 1 segundo
x = np.sin(2 * np.pi * f * t)  # señal senoidal
b = 3  # Tasa de bits por muestra
xRange = (-1, 1)
q = (xRange[1] - xRange[0]) / (2**b)  # Formula para el tamaño del cuantificador

# Añadir dithering a la señal cuantificada:
# Añadimos el cuantificador
qtz_ch = UniformSQ(b, xRange, qtype="midrise")

# Generar dither
dither_rectangular = genDither(len(x), q, pdf="rectangular")
dither_triangular = genDither(len(x), q, pdf="triangular")
dither_gaussian = genDither(len(x), q, pdf="gaussian")

# Representar histogramas de cada tipo de dither
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(dither_rectangular, bins=50, color="blue", alpha=0.7)
plt.title("Histograma - Dither Rectangular")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")

plt.subplot(1, 3, 2)
plt.hist(dither_triangular, bins=50, color="green", alpha=0.7)
plt.title("Histograma - Dither Triangular")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")

plt.subplot(1, 3, 3)
plt.hist(dither_gaussian, bins=50, color="red", alpha=0.7)
plt.title("Histograma - Dither Gaussiano")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

# Añadir dither a la señal
x_dither_uniform = x + dither_rectangular
x_dither_triangular = x + dither_triangular
x_dither_gaussian = x + dither_gaussian

# Cuantificar la señal con dither
xq_dither_uniform = qtz_ch.quantize(x_dither_uniform)
xq_dither_triangular = qtz_ch.quantize(x_dither_triangular)
xq_dither_gaussian = qtz_ch.quantize(x_dither_gaussian)

# Potencia del ruido agregado teorica y experimental
P_ruido_uniform_exp = np.mean(dither_rectangular**2)
P_ruido_triangular_exp = np.mean(dither_triangular**2)
P_ruido_gaussian_exp = np.mean(dither_gaussian**2)

P_ruido_uniform = q**2 / 12
P_ruido_triangular = q**2 / 6
P_ruido_gaussian = q**2 / 4

# Potencia del ruido total (dither + cuantificación)
P_ruido_total_uniform = q**2 / 6
P_ruido_total_triangular = q**2 / 4
P_ruido_total_gaussian = q**2 / 3


# Potencias dither (solo dither) en dB (teórica y experimental)
P_ruido_uniform_dB = 10 * np.log10(P_ruido_uniform)
P_ruido_triangular_dB = 10 * np.log10(P_ruido_triangular)
P_ruido_gaussian_dB = 10 * np.log10(P_ruido_gaussian)

P_ruido_uniform_exp_dB = 10 * np.log10(P_ruido_uniform_exp)
P_ruido_triangular_exp_dB = 10 * np.log10(P_ruido_triangular_exp)
P_ruido_gaussian_exp_dB = 10 * np.log10(P_ruido_gaussian_exp)

# Potencias totales (dither + cuantificación) en dB (teóricas)
P_ruido_total_uniform_dB = 10 * np.log10(P_ruido_total_uniform)
P_ruido_total_triangular_dB = 10 * np.log10(P_ruido_total_triangular)
P_ruido_total_gaussian_dB = 10 * np.log10(P_ruido_total_gaussian)

# Calcular la potencia del error experimental
P_e_dither_uniform = np.var((x - xq_dither_uniform))
P_e_dither_triangular = np.var((x - xq_dither_triangular))
P_e_dither_gaussian = np.var((x - xq_dither_gaussian))

# Cálculo en dB
P_e_dB_dither_uniform = 10 * np.log10(P_e_dither_uniform)
P_e_dB_dither_triangular = 10 * np.log10(P_e_dither_triangular)
P_e_dB_dither_gaussian = 10 * np.log10(P_e_dither_gaussian)

# Mostrar resultados: Potencia del ruido agregado (solo dither)
print("🔹 Potencia del ruido agregado (solo dither):")
print(
    f"  Teórica uniforme:   {P_ruido_uniform_dB:.2f} dB\t Experimental: {P_ruido_uniform_exp_dB:.2f} dB"
)
print(
    f"  Teórica triangular: {P_ruido_triangular_dB:.2f} dB\t Experimental: {P_ruido_triangular_exp_dB:.2f} dB"
)
print(
    f"  Teórica gaussiana:  {P_ruido_gaussian_dB:.2f} dB\t Experimental: {P_ruido_gaussian_exp_dB:.2f} dB\n"
)

# Mostrar resultados: Potencia del ruido total (dither + cuantificación)
print("🔸 Potencia del ruido total (dither + cuantificación):")
print(
    f"  Uniforme:   Teórica: {P_ruido_total_uniform_dB:.2f} dB\t Experimental: {P_e_dB_dither_uniform:.2f} dB"
)
print(
    f"  Triangular: Teórica: {P_ruido_total_triangular_dB:.2f} dB\t Experimental: {P_e_dB_dither_triangular:.2f} dB"
)
print(
    f"  Gaussiano:  Teórica: {P_ruido_total_gaussian_dB:.2f} dB\t Experimental: {P_e_dB_dither_gaussian:.2f} dB\n"
)

# Calcular la SNR para cada caso (no lo pide pero puede resultar interesante)
snr_dither_uniform = snr(x, xq_dither_uniform)
snr_dither_triangular = snr(x, xq_dither_triangular)
snr_dither_gaussian = snr(x, xq_dither_gaussian)

print("🔸 Relación señal ruido de la señal original y las cuatizadas:")
print(f"  SNR con dither uniforme: {snr_dither_uniform} dB")
print(f"  SNR con dither triangular: {snr_dither_triangular} dB")
print(f"  SNR con dither gaussiano: {snr_dither_gaussian} dB")


# Representar la señal original y la cuantificada con dither
plt.figure()
plt.plot(x[:160], color="red")
plt.title("Señal original")

plt.figure()
plt.plot(xq_dither_uniform[:160], color="red")
plt.title("Señal cuantificada \n con dither uniforme")

plt.figure()
plt.plot(xq_dither_triangular[:160], color="red")
plt.title("Señal cuantificada \n con dither triangular")

plt.figure()
plt.plot(xq_dither_gaussian[:160], color="red")
plt.title("Señal cuantificada \n con dither gaussiano")

plt.show()


# --- FFT y espectro dB para xq_dither_uniform ---
Xq_uniform = np.fft.fft(xq_dither_uniform)
freq_uniform = np.fft.fftfreq(len(xq_dither_uniform), d=1 / fs)
Xq_uniform = np.fft.fftshift(Xq_uniform)
freq_uniform = np.fft.fftshift(freq_uniform)

plt.figure()
plt.plot(freq_uniform, 10 * np.log10(np.abs(Xq_uniform) ** 2))
plt.title("Espectro - Dither uniforme")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()

# --- FFT y espectro dB para xq_dither_triangular ---
Xq_triangular = np.fft.fft(xq_dither_triangular)
freq_triangular = np.fft.fftfreq(len(xq_dither_triangular), d=1 / fs)
Xq_triangular = np.fft.fftshift(Xq_triangular)
freq_triangular = np.fft.fftshift(freq_triangular)

plt.figure()
plt.plot(freq_triangular, 10 * np.log10(np.abs(Xq_triangular) ** 2))
plt.title("Espectro - Dither triangular")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()

# --- FFT y espectro dB para xq_dither_gaussian ---
Xq_gaussian = np.fft.fft(xq_dither_gaussian)
freq_gaussian = np.fft.fftfreq(len(xq_dither_gaussian), d=1 / fs)
Xq_gaussian = np.fft.fftshift(Xq_gaussian)
freq_gaussian = np.fft.fftshift(freq_gaussian)

plt.figure()
plt.plot(freq_gaussian, 10 * np.log10(np.abs(Xq_gaussian) ** 2))
plt.title("Espectro - Dither gaussiano")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid()

plt.show()

# %% Ejercicio 4: Comparamos cuantización de media contrahuella con y sin dither para b = 3, 5
# Cargamos señal de audio:
fs, s = wf.read("Datos/altura.wav")
t = np.arange(len(s)) / fs

wav_range = signalRange(s)

# Cuantificamos la señal de audio con un cuantificador de media contrahuella de 3 bits por muestra
b = 3
qtz = UniformSQ(b, wav_range, qtype="midrise")
sq = qtz.quantize(s)
q = (wav_range[1] - wav_range[0]) / (2**b)

qtz_ch = UniformSQ(b, xRange, qtype="midrise")

dither_triangular = genDither(len(s), q, pdf="triangular")

sq = qtz.quantize(s)

s_dither = s + dither_triangular

sq_dither_3b = qtz.quantize(s_dither)

# Reproducimos la señal cuantificada sin dither
sd.play(sq, fs)
sd.wait()
sd.stop()

# Reproducimos la señal cuantificada con dither

sd.play(sq_dither_3b, fs)
sd.wait()
sd.stop()

# Calculamos la SNR de las dos señales cuantificadas
snr_sq = snr(s, sq)
snr_sq_dither = snr(s, sq_dither_3b)

print(f"SNR de la señal cuantificada con 3 bits: {snr_sq} dB")
print(f"SNR de la señal cuantificada con dither con 3 bits: {snr_sq_dither} dB")

b = 5
qtz = UniformSQ(b, wav_range, qtype="midrise")
sq = qtz.quantize(s)
q = (wav_range[1] - wav_range[0]) / (2**b)

qtz_ch = UniformSQ(b, xRange, qtype="midrise")

dither_triangular = genDither(len(s), q, pdf="triangular")

sq = qtz.quantize(s)

s_dither = s + dither_triangular

sq_dither_3b = qtz.quantize(s_dither)

# Reproducimos la señal cuantificada sin dither
sd.play(sq, fs)
sd.wait()
sd.stop()

# Reproducimos la señal cuantificada con dither

sd.play(sq_dither_3b, fs)
sd.wait()
sd.stop()

# Calculamos la SNR de las dos señales cuantificadas
snr_sq = snr(s, sq)
snr_sq_dither = snr(s, sq_dither_3b)

print(f"SNR de la señal cuantificada con 5 bits: {snr_sq} dB")
print(f"SNR de la señal cuantificada con dither con 5 bits: {snr_sq_dither} dB")

# %% EJercicio 5:
# Abrimos lena.png, cuantificamos añadimos dither y calculamos.
# Cargamos la imagen de Lena en escala de grises
img = io.imread("Datos/lena.png")

img = color.rgb2gray(img)

# Rango de valores de la imagen
img_range = signalRange(img)

# Cuantificación de la imagen con un cuantificador de media contrahuella de 3 bits por muestra
b = 3
qtz = UniformSQ(b, img_range, qtype="midrise")

imgq = qtz.quantize(img)

q = (img_range[1] - img_range[0]) / (2**b)

# Añadir dither triangular a la imagen
dither_triangular = genDither(img.shape, q / 5, pdf="triangular")

img_dither = img + dither_triangular
imgq_dither = qtz.quantize(img_dither)

# Calcular la SNR de las dos imágenes cuantificadas
snr_img = snr(img, imgq)
snr_img_dither = snr(img, imgq_dither)

print(f"SNR de la imagen cuantificada: {snr_img} dB")
print(f"SNR de la imagen cuantificada con dither: {snr_img_dither} dB")

# Representar las imágenes en gráficos separados
plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.show()

plt.figure()
plt.imshow(imgq, cmap="gray")
plt.title("Cuantificada")
plt.show()

plt.figure()
plt.imshow(imgq_dither, cmap="gray")
plt.title("Cuantificada con dither")
plt.show()

# %% Ejercicio 6
# Algoritmo de Floyd-Steinberg

img = io.imread("Datos/lena.png")
img_range = signalRange(img)
b = 3
qtz = UniformSQ(b, img_range, qtype="midrise")

img_dithered = np.zeros_like(img)
img_qtz = np.zeros_like(img)

for i in range(img.shape[0]):  # Para cada fila
    for j in range(img.shape[1]):  # Para cada columna
        for k in range(img.shape[2]):  # Para cada canal de color (RGB)
            img_qtz[i, j, k] = qtz.quantize(img[i, j, k])  # i fila, j columna y k canal

            eps = np.float32(img[i, j, k]) - np.float32(img_qtz[i, j, k])

            img_dithered[i, j, k] += img_qtz[i, j, k]

            # distribuimos si se puede el error a los pixeles vecinos
            if j + 1 < img.shape[1]:
                img_dithered[i, j + 1, k] += eps * (7 / 16)

            if i + 1 < img.shape[0] and j - 1 >= 0:
                img_dithered[i + 1, j - 1, k] += eps * (3 / 16)

            if i + 1 < img.shape[0]:
                img_dithered[i + 1, j, k] += eps * (5 / 16)

            if i + 1 < img.shape[0] and j + 1 < img.shape[1]:
                img_dithered[i + 1, j + 1, k] += eps * (1 / 16)

# Representamos
plt.figure()
plt.imshow(img)
plt.title("Original")
plt.show()

plt.figure()
plt.imshow(img_qtz)
plt.title("Cuantificada")
plt.show()

plt.figure()
plt.imshow(img_dithered)
plt.title("Cuantificada con dither")
plt.show()

# Calcular la SNR de las dos imágenes cuantificadas
snr_img = snr(img, img_qtz)
snr_img_dither = snr(img, img_dithered)

print(f"SNR de la imagen cuantificada: {snr_img} dB")
print(f"SNR de la imagen cuantificada con dither: {snr_img_dither} dB")
