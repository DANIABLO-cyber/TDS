import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scalib import UniformSQ, genDither

# === Cargar imagen lena en escala de grises ===
img = Image.open(".../Datos/lena.png").convert("L")  # modo 'L' = escala de grises
img_np = np.asarray(img).astype(np.float32) / 255.0  # normalizar entre 0 y 1

# === Parámetros de cuantificación ===
b = 3
xRange = (0, 1)
qtz = UniformSQ(b, xRange, qtype="midrise")
q = (xRange[1] - xRange[0]) / (2**b)  # cuanto

# === Cuantificación sin dither ===
img_q = qtz.quantize(img_np)

# === Cuantificación con dither triangular ===
dither_triangular = genDither(img_np.size, "triangular", q / 5).reshape(img_np.shape)
img_dith = img_np + dither_triangular
img_q_dith = qtz.quantize(img_dith)


# === Cálculo de SNR (Señal vs Error de cuantificación) ===
def snr(original, reconstruida):
    error = original - reconstruida
    power_signal = np.mean(original**2)
    power_error = np.mean(error**2)
    return 10 * np.log10(power_signal / power_error)


snr_sin_dither = snr(img_np, img_q)
snr_con_dither = snr(img_np, img_q_dith)

# === Mostrar imágenes ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap="gray", vmin=0, vmax=1)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_q, cmap="gray", vmin=0, vmax=1)
plt.title(f"Cuantificada sin Dither\nSNR = {snr_sin_dither:.2f} dB")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_q_dith, cmap="gray", vmin=0, vmax=1)
plt.title(f"Cuantificada con Dither Triangular\nSNR = {snr_con_dither:.2f} dB")
plt.axis("off")

plt.tight_layout()
plt.show()

# === Explicación del resultado ===
print("Observación:")
print(
    "- La imagen cuantificada sin dither tiene mayor SNR pero más 'banding' (zonas planas artificiales)."
)
print(
    "- La imagen con dither tiene un SNR ligeramente menor debido al ruido añadido, pero el resultado visual es más natural."
)
print(
    "- El dither suaviza transiciones, evitando artefactos visibles causados por la cuantificación."
)
