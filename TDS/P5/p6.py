# Práctica 6: Procesamiento digital de imágenes
# Autor: Guillermo Ruvira Quesada y Miguel Ángel Parrilla

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
import scipy.ndimage as ndi
from scipy import signal

# %%


def ecualizar(imagen, niveles=10):
    hist, _ = np.histogram(imagen, bins=np.arange(niveles + 1))
    H_acum = np.cumsum(hist)
    H_norm = H_acum / H_acum[-1] * (niveles - 1)
    imagen_eq = np.zeros_like(imagen)
    for i in range(niveles):
        imagen_eq[imagen == i] = H_norm[i]

    return imagen_eq, hist, H_norm


# %%

I = io.imread("tire.tif")  # Lectura de imagen tiff

Nbins = 256  # Número de niveles
bins = np.arange(-0.5, Nbins + 0.5)  # Definir límites de los subintervalos
Hist, bins = np.histogram(I, bins)
centers = 0.5 * (bins[1:] + bins[:-1])  # Calcular centros de los subintervalos

# Usar la función ecualizar
I_int = I.astype(int)
I_eq, Hist_eq, H_norm = ecualizar(I_int, niveles=Nbins)

# Histograma de la imagen ecualizada
Hist_eq2, bins_eq = np.histogram(I_eq, bins)
centers_eq = 0.5 * (bins_eq[1:] + bins_eq[:-1])

# Subplot de imágenes
fig_img, axs_img = plt.subplots(1, 2, figsize=(10, 5))
axs_img[0].imshow(I, cmap="gray")
axs_img[0].set_title("Imagen Original")
axs_img[0].axis("off")
axs_img[1].imshow(I_eq, cmap="gray")
axs_img[1].set_title("Imagen Ecualizada")
axs_img[1].axis("off")
plt.tight_layout()

# Subplot de histogramas
fig_hist, axs_hist = plt.subplots(1, 3, figsize=(15, 4))
axs_hist[0].stem(centers, Hist, basefmt=" ")
axs_hist[0].set_title("Histograma Original")
axs_hist[1].stem(centers, H_norm, basefmt=" ")
axs_hist[1].set_title("Histograma Acumulado Normalizado")
axs_hist[2].stem(centers_eq, Hist_eq2, basefmt=" ")
axs_hist[2].set_title("Histograma Imagen Ecualizada")
for ax in axs_hist:
    ax.set_xlim([0, Nbins - 1])
plt.tight_layout()
plt.show()

# %%
# Crear imagen artificial ejemplo para ver cómo funciona la ecualización de histograma

oscura = np.array([[0, 1, 1, 0, 1], [1, 2, 1, 0, 1], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0]])

clara = np.array([[8, 9, 9, 8, 9], [9, 9, 8, 9, 8], [8, 8, 9, 9, 8], [9, 8, 8, 9, 8]])

normal = np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [0, 1, 4, 6, 8], [2, 3, 5, 7, 9]])

imagenes = [oscura, clara, normal]
titulos = ["Oscura", "Clara", "Normal"]

# ======== Procesar imágenes =========
resultados = []
for img in imagenes:
    eq, hist, H_norm = ecualizar(img)
    hist_eq, _ = np.histogram(eq, bins=np.arange(11))
    resultados.append(
        {
            "original": img,
            "ecualizada": eq,
            "hist": hist,
            "H_norm": H_norm,
            "hist_eq": hist_eq,
        }
    )

# ======== Gráfico 1: Histogramas 3x3 =========
fig1, axs1 = plt.subplots(3, 3, figsize=(12, 9))
for i, res in enumerate(resultados):
    axs1[i, 0].bar(np.arange(10), res["hist"], color="gray", edgecolor="black")
    axs1[i, 0].set_title(f"{titulos[i]} - Hist. Original")

    axs1[i, 1].bar(np.arange(10), res["H_norm"], color="blue", edgecolor="black")
    axs1[i, 1].set_title(f"{titulos[i]} - Hist. Acumulado Norm.")

    axs1[i, 2].bar(np.arange(10), res["hist_eq"], color="green", edgecolor="black")
    axs1[i, 2].set_title(f"{titulos[i]} - Hist. Ecualizado")

    for j in range(3):
        axs1[i, j].set_xticks(np.arange(10))
        axs1[i, j].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# ======== Gráfico 2: Imágenes 3x2 =========
fig2, axs2 = plt.subplots(3, 2, figsize=(8, 9))
for i, res in enumerate(resultados):
    axs2[i, 0].imshow(res["original"], cmap="gray", vmin=0, vmax=9)
    axs2[i, 0].set_title(f"{titulos[i]} - Original")
    axs2[i, 0].axis("off")

    axs2[i, 1].imshow(res["ecualizada"], cmap="gray", vmin=0, vmax=9)
    axs2[i, 1].set_title(f"{titulos[i]} - Ecualizada")
    axs2[i, 1].axis("off")

plt.tight_layout()
plt.show()
# concluimos que la ecualizador del histograma nos sirve para repartir de manera eficiente
# los niveles de gris en la imagen, mejorando el contraste y la visibilidad de detalles.
# por que, si nos fijamos en la imagen original, los grises se reparten en funcion de lo oscura o clara que sea la imagen
# los terminos que mas contribuyen al histograma seran los que creen mas escalones (estaran mas diferenciados)

# %% Transformada de Fourier

# Transformada de Fourier y espectro
# Cargar imagen y convertir a niveles de gris
I = io.imread("mandrill.tif", as_gray=True)
plt.figure(1)
plt.subplot(1, 2, 1), plt.imshow(I, cmap="gray")
# Se selecciona una subimagen con una textura clara
B = I[130 : 130 + 100, 150 : 150 + 200]
plt.subplot(1, 2, 2), plt.imshow(B, cmap="gray")
N, M = B.shape
imsize = N * M
plt.figure(2)
plt.imshow(B, cmap="gray")

# Computar FFT2
F = np.fft.fft2(B)
# Usar escala logaritmica para correcta visualizacion
plt.imshow(10 * np.log10(np.abs(F) ** 2 / imsize), cmap="hot")
plt.colorbar()
# Cuadrantes reordenados (frec. (0,0) en el centro)
F = np.fft.fftshift(F)
plt.figure(3)
plt.imshow(10 * np.log10(np.abs(F) ** 2 / imsize), cmap="hot")
plt.colorbar()
plt.show()

# las frecuencias bajas son las transiciones lentas de la imagen, mientras que las altas son los detalles finos

# %% Restauración de imágenes
# FILTRADO LINEAL 2D SEPARABLE Y FILTRO DE MEDIANA

I = io.imread("lena.bmp", as_gray=True)  # Cargar imagen
J = util.random_noise(I, mode="s&p")  # Contaminar con ruido sal y pimienta

Lfir = 11  # Orden FIR (impar)
fc = 0.18  # Frecuencia de corte (fs=1 --> frecuencia de muestreo)
Fs = 1  # Frecuencia de muestreo
# Obtener filtro (metodo ventana; ventana Kaiser por defecto)
h_1D = signal.firwin(Lfir, fc, fs=Fs)

# Filtro 2D separable
h_2D = np.outer(h_1D, h_1D)

# Representar filtro 2D separable y resultados en subplots 2x2

fig_fir, axs_fir = plt.subplots(2, 2, figsize=(10, 10))

# Arriba: Imagen original y con ruido
axs_fir[0, 0].imshow(I, cmap="gray")
axs_fir[0, 0].set_title("Imagen Original")
axs_fir[0, 0].axis("off")

axs_fir[0, 1].imshow(J, cmap="gray")
axs_fir[0, 1].set_title("Imagen con Ruido Sal y Pimienta")
axs_fir[0, 1].axis("off")

# Abajo: Filtro 2D separable y señal filtrada
axs_fir[1, 0].imshow(h_2D, cmap="gray")
axs_fir[1, 0].set_title("Filtro 2D Separable")
axs_fir[1, 0].axis("off")

J_filt = ndi.convolve(J, h_2D, mode="reflect")
axs_fir[1, 1].imshow(J_filt, cmap="gray")
axs_fir[1, 1].set_title("Imagen Filtrada (Filtro 2D Separable)")
axs_fir[1, 1].axis("off")

plt.tight_layout()
plt.show()

# Aplicar filtro de la mediana con diferentes tamaños de ventana
sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19]
J_meds = [ndi.median_filter(J, size=s) for s in sizes]

# Mostrar resultados de los filtros de mediana en una figura 3x3
fig_med, axs_med = plt.subplots(3, 3, figsize=(15, 15))
for idx, (ax, img, sz) in enumerate(zip(axs_med.flat, J_meds, sizes)):
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Filtro de Mediana {sz}x{sz}")
    ax.axis("off")
plt.tight_layout()
plt.show()
