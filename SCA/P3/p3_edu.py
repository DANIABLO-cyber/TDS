#!/usr/bin/env python3
# ejercicio1_pcm.py
# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scalib import UniformSQ, FixedLengthCoder, signalRange
# %%


def encoderPCM(data, dataRange, b):
    """
    Codifica la imagen `data` (2D) usando PCM:
      - UniformSQ para cuantificación de b bits
      - FixedLengthCoder para pasar ids a bits
    Devuelve:
      - code: array 1D de bits (0/1) empaquetados en bytes.
    """
    # 1) Mapeo del codificador: array de ids enteros [0, 2^b)
    qtz = UniformSQ(b, dataRange)
    ids = qtz.encode(data)  # shape = data.shape, dtype=int

    # 2) Codificador de longitud fija: ids → secuencia de bits
    coder = FixedLengthCoder(b)
    code = coder.encode(ids.flatten())
    return code


def decoderPCM(code, dataRange, b):
    """
    Descodifica la secuencia de bits `code` con PCM:
      - FixedLengthCoder para recuperar ids
      - UniformSQ.decode para reconstruir valores
    Devuelve:
      - data: array 1D de valores reconstruidos.
    """
    # 1) Descodificar bits → ids
    coder = FixedLengthCoder(
        b
    )  # Cada grupo de b bits lo convierte en un número entero (un id)
    ids = coder.decode(code)  # 1D array de enteros

    # 2) Mapeo del descodificador: ids → valores aproximados
    qtz = UniformSQ(b, dataRange)
    data = qtz.decode(ids)  # 1D array de floats
    return data


def calculate_snr(orig, recon):
    """
    SNR = 10 log10( var(orig) / var(orig - recon) )
    """
    noise = orig - recon
    return 10 * np.log10(np.var(orig) / np.var(noise))


def main():
    # --- Parámetros ---
    img_path = "Data/lena.png"  # Coloca lena.png en este directorio
    bit_rates = [2, 3, 4, 6, 8]  # Tasas de bits a probar

    # 1) Leer imagen
    im = img_as_float(io.imread(img_path, as_gray=True))
    h, w = im.shape

    # 2) Rango de valores para el cuantificador
    dataRange = signalRange(
        im
    )  # (min, max) de la imagen :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

    results = []

    # 3) Codificar / Decodificar para cada tasa de bits
    for b in bit_rates:
        # Codificación PCM
        code = encoderPCM(im, dataRange, b)

        # Decodificación PCM
        flat = decoderPCM(code, dataRange, b)
        recon = flat.reshape((h, w))

        # Calcular SNR
        snr = calculate_snr(im, recon)
        results.append((b, recon, snr))

    # 4) Visualizar original + reconstrucciones
    n = len(results) + 1
    plt.figure(figsize=(3 * n, 3))

    # Imagen original
    plt.subplot(1, n, 1)
    plt.imshow(im, cmap="gray", vmin=0, vmax=1)
    plt.title("Original")
    plt.axis("off")

    # Imágenes reconstruidas
    for i, (b, recon, snr) in enumerate(results, start=2):
        plt.subplot(1, n, i)
        plt.imshow(recon, cmap="gray", vmin=0, vmax=1)
        plt.title(f"{b} bits\nSNR={snr:.1f} dB")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# %%
# %%


def dct(N):
    """
    Calcula la matriz de transformación DCT de tamaño N x N según la ecuación:
    C[n,k] = sqrt(a[n]/N) * cos((2*k + 1) * n * pi / (2*N)),
    donde a[0] = 1, a[n>0] = 2.

    Parámetros:
    ----------
    N : int
        Tamaño de la DCT (número de filas/columnas).

    Devuelve:
    --------
    C : ndarray de forma (N, N)
        Matriz de transformación DCT.
    """
    # Vector de factores a[n]
    a = np.full(N, 2.0)
    a[0] = 1.0
    # Preparar matriz C
    C = np.zeros((N, N), dtype=float)
    # Constante común
    factor = np.pi / (2.0 * N)
    # Llenado de C
    for n in range(N):
        for k in range(N):
            C[n, k] = np.sqrt(a[n] / N) * np.cos((2 * k + 1) * n * factor)
    return C


def dDCT(block, C):
    """
    Aplica la DCT directa (bidimensional) a un bloque de datos usando la matriz C:
    Theta = C @ block @ C.T

    Parámetros:
    ----------
    block : ndarray de forma (N, N)
        Bloque de datos en el dominio espacial.
    C : ndarray de forma (N, N)
        Matriz de transformación DCT.

    Devuelve:
    --------
    coef : ndarray de forma (N, N)
        Matriz de coeficientes en el dominio de frecuencia.
    """
    return C @ block @ C.T


def iDCT(coef, C):
    """
    Aplica la DCT inversa (bidimensional) a la matriz de coeficientes:
    block = C.T @ coef @ C

    Parámetros:
    ----------
    coef : ndarray de forma (N, N)
        Coeficientes en el dominio de frecuencia.
    C : ndarray de forma (N, N)
        Matriz de transformación DCT.

    Devuelve:
    --------
    block : ndarray de forma (N, N)
        Bloque de datos recuperado en el dominio espacial.
    """
    return C.T @ coef @ C


if __name__ == "__main__":
    # Ejemplo de prueba
    N = 8
    C = dct(N)
    # Bloque de ejemplo: gradiente lineal
    block = np.arange(N * N).reshape(N, N).astype(float)
    theta = dDCT(block, C)
    rec = iDCT(theta, C)
    # Verificar reconstrucción
    print("Error máximo reconstrucción:", np.max(np.abs(block - rec)))


# %%
def encoderDCT(data, dataRange, N, Nsel, b):
    """
    Codifica la imagen `data` (2D) con DCT + muestreo zonal:
      - dataRange = (mín, máx) rango de píxeles de `data`
      - N = tamaño de bloque
      - Nsel = tamaño de la submatriz de coeficientes (zonales)
      - b = bits de cuantificación

    Devuelve:
      code : array 1D con la secuencia de bits codificados.
    """
    h, w = data.shape
    # Pad para que ambos divisibles por N
    pad_h = (N - h % N) % N
    pad_w = (N - w % N) % N
    padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode="constant")

    C = dct(N)
    coeffs = []

    # Extraer coeficientes zonales de cada bloque
    for i in range(0, padded.shape[0], N):
        for j in range(0, padded.shape[1], N):
            block = padded[i : i + N, j : j + N]
            theta = dDCT(block, C)
            coeffs.append(theta[:Nsel, :Nsel].ravel())

    coeffs = np.hstack(coeffs)

    # Cuantificar y codificar
    qtz = UniformSQ(b, dataRange)
    ids = qtz.encode(coeffs)
    coder = FixedLengthCoder(b)
    code = coder.encode(ids)
    return code


def decoderDCT(code, dataRange, N, Nsel, imageShape, b):
    """
    Decodifica la secuencia `code` producida por encoderDCT:
      - dataRange, N, Nsel, b igual que en encoder
      - imageShape = (alto, ancho) original de la imagen

    Devuelve:
      data : array 2D reconstruido en el rango original.
    """
    # Decodificar bits y descuantificar
    coder = FixedLengthCoder(b)
    ids = coder.decode(code)
    qtz = UniformSQ(b, dataRange)
    coeffs = qtz.decode(ids)

    # Número de bloques
    blocks_h = (imageShape[0] + N - 1) // N
    blocks_w = (imageShape[1] + N - 1) // N
    total = blocks_h * blocks_w
    coeffs = coeffs.reshape((total, Nsel, Nsel))

    C = dct(N)
    padded_h = blocks_h * N
    padded_w = blocks_w * N
    recon = np.zeros((padded_h, padded_w))

    idx = 0
    for i in range(0, padded_h, N):
        for j in range(0, padded_w, N):
            theta = np.zeros((N, N))
            theta[:Nsel, :Nsel] = coeffs[idx]
            block = iDCT(theta, C)
            recon[i : i + N, j : j + N] = block
            idx += 1

    # Recortar al tamaño original
    return recon[: imageShape[0], : imageShape[1]]


# %%


# Se asume que encoderDCT, decoderDCT, encoderPCM, decoderPCM y calculate_snr están definidos o importados


def main():
    # --- 1) Carga y preprocesado ---
    im = img_as_float(io.imread("lena.png", as_gray=True))
    # original en [0,1]; lo centramos en 0 → [-0.5, +0.5]
    im0 = im - 0.5
    # valor máximo global de im0
    max_val_global = np.max(np.abs(im0))
    shape = im0.shape

    # --- 2) Parámetros a probar ---
    bloque_sizes = [8, 16]
    bit_rates = [2, 4, 8]

    # --- 3) Preparar figura ---
    fig, axes = plt.subplots(
        len(bloque_sizes),
        len(bit_rates),
        figsize=(3 * len(bit_rates), 3 * len(bloque_sizes)),
    )
    plt.gray()

    # --- 4) Recorremos combinaciones ---
    for i, N in enumerate(bloque_sizes):
        # Ajustar el rango dinámico del cuantificador según N
        data_range = (-N * max_val_global, N * max_val_global)

        for j, b in enumerate(bit_rates):
            Nsel = N // 2

            # 4.1) DCT + zonal
            code_dct = encoderDCT(im0, data_range, N, Nsel, b)
            rec_dct = decoderDCT(code_dct, data_range, N, Nsel, shape, b) + 0.5
            snr_dct = calculate_snr(im, rec_dct)
            size_dct = len(code_dct) / 8 / 1024

            # 4.2) PCM uniforme
            code_pcm = encoderPCM(im0, data_range, b)
            rec_pcm = decoderPCM(code_pcm, data_range, b).reshape(shape) + 0.5
            snr_pcm = calculate_snr(im, rec_pcm)
            size_pcm = len(code_pcm) / 8 / 1024

            # 4.3) Mostrar reconstrucción DCT
            ax = axes[i, j]
            ax.imshow(rec_dct, vmin=0, vmax=1)
            ax.set_title(
                f"N={N}, b={b}\n"
                f"DCT: SNR={snr_dct:.1f}dB, {size_dct:.1f}kB\n"
                f"PCM: SNR={snr_pcm:.1f}dB, {size_pcm:.1f}kB"
            )
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
# %%
