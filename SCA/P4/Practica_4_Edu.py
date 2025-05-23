# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# --- Codificador PCM Adaptable (idéntico al tuyo) ---
def encoderAPCM(data, wlen, dataRange, b, bg):
    g_max = max(abs(dataRange[0]), abs(dataRange[1]))
    quant_ranges = {
        b: (-1.0, 1.0),
        bg: (0.0, g_max),
    }

    def Qencode(bits, values):
        vmin, vmax = quant_ranges[bits]
        levels = 2**bits
        step = (vmax - vmin) / levels
        idx = np.floor((values - vmin) / step).astype(int)
        return np.clip(idx, 0, levels - 1)

    def Cencode(bits, indices):
        idxs = np.atleast_1d(indices).astype(int)
        bitstream = []

        for idx in idxs:
            # format(idx, '0Nb') devuelve la cadena binaria de longitud N, rellenada con ceros.
            bin_str = format(idx, f"0{bits}b")

            for b in bin_str:
                bitstream.append(int(b))

        return np.array(bitstream, dtype=np.uint8)

    bitstream = []
    for k in range(0, len(data), wlen):
        w = data[k : k + wlen]
        if len(w) < wlen:
            break
        g = np.max(np.abs(w)) or 1.0
        w_norm = w / g
        i_w = Qencode(b, w_norm)
        i_g = Qencode(bg, np.array([g]))
        bitstream.extend(Cencode(bg, i_g).tolist())
        bitstream.extend(Cencode(b, i_w).tolist())
    return np.array(bitstream, dtype=np.uint8)


# --- Decodificador PCM Adaptable (corregido) ---
def decoderAPCM(code, wlen, dataRange, b, bg):
    """
    code      : array de bits (uint8)
    wlen      : tamaño de ventana (ej. 120)
    dataRange : (min, max) de las muestras originales
    b         : bits/muestra para la señal
    bg        : bits/muestra para la ganancia
    """
    # Parámetros de cuantización
    g_max = max(abs(dataRange[0]), abs(dataRange[1]))
    quant_ranges = {
        b: (-1.0, 1.0),
        bg: (0.0, g_max),
    }

    # Decodifica un vector de bits a índices
    def Cdecode(bits, bits_seq):
        arr = np.array(bits_seq, dtype=int)
        arr = arr.reshape(-1, bits)
        # Estas líneas convierten la
        # secuencia plana de bits en una
        # matriz donde cada fila contiene
        # los bits de un índice cuantizado,
        # listos para ser decodificados a su valor decimal.
        indices = []
        for row in arr:
            idx = 0
            for bit in row:
                idx = (idx << 1) | bit  # Desplaza a la izquierda y añade el bit
                # El operador | (OR a nivel de bits) en
                # la expresión idx = (idx << 1) | bit añade
                # el nuevo bit en la posición menos
                # significativa después de desplazar
                # el número anterior a la izquierda.

            indices.append(idx)
        return np.array(indices)

    # Reconstruye valor(es) a partir de índices
    def Qdecode(bits, indices):
        vmin, vmax = quant_ranges[bits]
        levels = 2**bits
        step = (vmax - vmin) / levels
        return vmin + (indices.astype(float) + 0.5) * step

    bits_per_win = bg + b * wlen
    Nw = len(code) // bits_per_win  # número de ventanas completas codificadas

    out = []  # iremos concatenando cada ventana reconstruida
    for i in range(Nw):
        base = i * bits_per_win
        # ganancia
        c_g = code[base : base + bg]
        i_g = Cdecode(bg, c_g.reshape(1, -1))
        g = Qdecode(bg, i_g)[0]
        # señal normalizada
        if b > 0:
            c_w = code[base + bg : base + bg + b * wlen]
            i_w = Cdecode(b, c_w.reshape(wlen, -1))
            w_norm = Qdecode(b, i_w)
        else:
            w_norm = np.zeros(wlen, dtype=float)
        # reconstrucción de la ventana
        out.append(w_norm * g)

    if len(out):
        return np.concatenate(out)
    else:
        return np.array([], dtype=float)


# --- Función SNR y PCM estándar (sin cambios) ---
def compute_snr(orig, recon):
    L = min(len(orig), len(recon))
    noise = orig[:L] - recon[:L]
    return 10 * np.log10(np.sum(orig[:L] ** 2) / np.sum(noise**2))


def pcm_standard(data, b, dataRange):
    vmin, vmax = dataRange
    levels = 2**b
    step = (vmax - vmin) / levels
    idx = np.floor((data - vmin) / step).astype(int)
    idx = np.clip(idx, 0, levels - 1)
    return vmin + (idx + 0.5) * step


# --- Carga y normalización ---
fs, data = wavfile.read("marlene.wav")
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768
elif data.dtype == np.int32:
    data = data.astype(np.float32) / 2147483648
elif data.dtype == np.uint8:
    data = (data.astype(np.float32) - 128) / 128
else:
    data = data.astype(np.float32)
dataRange = (data.min(), data.max())

# --- Parámetros y cálculo de SNRs ---
wlen = 120
bg = 16
b_vals = [1, 2, 3, 4, 5, 6, 8]
snr_apcm = []
snr_pcm = []

for b in b_vals:
    code = encoderAPCM(data, wlen, dataRange, b, bg)
    rec = decoderAPCM(code, wlen, dataRange, b, bg)
    snr_apcm.append(compute_snr(data, rec))
    rec_std = pcm_standard(data, b, dataRange)
    snr_pcm.append(compute_snr(data, rec_std))

# --- Gráfica corregida ---
plt.figure(figsize=(8, 5))
plt.plot(b_vals, snr_apcm, "o-", label="APCM")
plt.plot(b_vals, snr_pcm, "s--", label="PCM estándar")
plt.xlabel("Bits por muestra (b)")
plt.ylabel("SNR [dB]")
plt.title("Comparación de SNR: PCM Adaptable vs PCM Estándar")
plt.grid(True)
plt.legend()
plt.show()

# %%

# EJERCICIO 2

import numpy as np
from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt


def compute_qmf_filters(numtaps, cutoff):
    """
    Calcula los coeficientes de los 4 filtros QMF para codificación en sub-bandas de 2 canales.
    numtaps: número de coeficientes (orden del filtro + 1)
    cutoff: frecuencia de corte normalizada (0.5 corresponde a 0.5·π rad/muestra)
    Devuelve: h1, h2, k1, k2
    """
    # Filtro paso–baja de análisis
    h1 = firwin(numtaps, cutoff, window="hamming")
    n = np.arange(numtaps)
    # Filtro paso–alta de análisis (espejo en cuadratura)
    h2 = ((-1) ** n) * h1
    # Filtro paso–baja de síntesis
    k1 = 2 * h1
    # Filtro paso–alta de síntesis
    k2 = -2 * h2
    return h1, h2, k1, k2


# Parámetros de ejemplo
numtaps = 24
cutoff = 0.5

# Cálculo de coeficientes
h1, h2, k1, k2 = compute_qmf_filters(numtaps, cutoff)

# Mostrar coeficientes
print("Coeficientes de h1:", h1)
print("Coeficientes de h2:", h2)
print("Coeficientes de k1:", k1)
print("Coeficientes de k2:", k2)

# Graficar respuestas en frecuencia
filters = {
    "h1 (Analysis LPF)": h1,
    "h2 (Analysis HPF)": h2,
    "k1 (Synthesis LPF)": k1,
    "k2 (Synthesis HPF)": k2,
}

for name, coeffs in filters.items():
    w, H = freqz(coeffs, worN=1024)
    plt.figure()
    plt.title(f"Respuesta en frecuencia de {name}")
    plt.plot(w / np.pi, 20 * np.log10(np.abs(H)))
    plt.xlabel("Frecuencia normalizada (×π rad/muestra)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True)

plt.show()


# %%
# EJERCICIO 3


def overall_response(h1, h2, k1, k2):
    """Convolución y suma según g(n) = (h1*k1 + h2*k2)/2"""
    g1 = np.convolve(h1, k1)
    g2 = np.convolve(h2, k2)
    g = 0.5 * (g1 + g2)
    return g


def plot_response(g, label):
    w, H = freqz(g, worN=2048)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(H)), label=label)


def compute_and_plot(case_label, h1):
    numtaps = len(h1)
    n = np.arange(numtaps)
    h2 = ((-1) ** n) * h1
    k1 = 2 * h1
    k2 = -2 * h2
    g = overall_response(h1, h2, k1, k2)
    delay = (len(g) - 1) / 2
    print(f"{case_label}: longitud g = {len(g)}, retardo = {delay:.1f} muestras")
    plot_response(g, case_label)


# Preparar figura
plt.figure(figsize=(8, 5))

# 1) Caso original: cutoff = 0.5
numtaps = 24
cutoff1 = 0.5
h1_orig, _, _, _ = compute_qmf_filters(numtaps, cutoff1)
compute_and_plot("Cutoff 0.50×Nyquist", h1_orig)

# 2) Caso cutoff + 7%
cutoff2 = cutoff1 + 0.025
h1_high, _, _, _ = compute_qmf_filters(numtaps, cutoff2)
compute_and_plot(f"Cutoff {cutoff2:.3f}×Nyquist", h1_high)

# 3) LPF G.722 (si scalib está disponible)
try:
    import scalib

    h1_g722 = scalib.getG722lpf()
    compute_and_plot("LPF G.722", h1_g722)
except ImportError:
    print("No se encontró 'scalib' en el entorno. Instala scalib para el caso G.722.")

# Configuración final del gráfico
plt.title("Respuesta en frecuencia del sistema completo g(n)")
plt.xlabel("Frecuencia normalizada (×π rad/muestra)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.show()
# %%

# EJERCICIO 4

import numpy as np
from scalib import getG722lpf


def getBitsPerChannel(nBits, wlen, bLB, bHB, bG):
    bGLB = bG if bLB > 0 else 0
    bGHB = bG if bHB > 0 else 0
    wSize = wlen * (bLB + bHB) + bGLB + bGHB
    nSamples = nBits // wSize * wlen + max(nBits % wSize - bGLB - bGHB, 0) // (
        bLB + bHB
    )
    nLB = int(nSamples * bLB + np.ceil(nSamples / wlen) * bGLB)
    nHB = nBits = nLB
    return nLB, nHB


def encoderG722(data, wlen, dataRange, bl, bh, bg):
    """
    data      : 1D numpy array con la señal de audio
    wlen      : tamaño de ventana (ej. 120)
    dataRange : (min, max) posibles de las muestras
    bl, bh    : bits/muestra para canal baja y alta
    bg        : bits/muestra para las ganancias
    retorna   : 1D numpy array con la secuencia de bits codificados
    """
    # 1) construye filtros de análisis
    h1 = getG722lpf()  # h1(n): LPF :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    n = np.arange(len(h1))
    h2 = (
        ((-1) ** n) * h1
    )  # h2(n) = (−1)^n h1(n) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

    # 2) filtra toda la señal
    sl = np.convolve(data, h1, mode="same")
    sh = np.convolve(data, h2, mode="same")

    # 3) reduce muestreo a la mitad
    sl_ds = sl[::2]
    sh_ds = sh[::2]

    # 4) codifica cada subbanda con tu APCM
    cl = encoderAPCM(sl_ds, wlen, dataRange, bl, bg)
    ch = encoderAPCM(sh_ds, wlen, dataRange, bh, bg)

    # 5) concatena los bits de baja y alta
    return np.concatenate([cl, ch])


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# —————— Decoder G.722 usando decoderAPCM nuevo ——————
from scalib import getG722lpf  # o tu propia función que devuelve 24 coeficientes


def decoderG722(code, wlen, dataRange, bl, bh, bg):
    # 1) filtros de síntesis
    h1 = getG722lpf()
    k1 = 2 * h1
    n = np.arange(len(h1))
    k2 = -2 * ((-1) ** n) * h1

    # 2) calcular bits por ventana para cada sub-banda
    bits_low = bg + bl * wlen
    bits_high = bg + bh * wlen
    # 3) partir stream global en bajo/alto
    Nw = len(code) // (bits_low + bits_high)
    nLB = Nw * bits_low
    cl = code[:nLB]
    ch = code[nLB : nLB + Nw * bits_high]

    # 4) decodificar cada rama
    sl_ds = decoderAPCM(cl, wlen, dataRange, bl, bg)
    sh_ds = decoderAPCM(ch, wlen, dataRange, bh, bg)

    # 5) sobremuestrear e intercalar ceros
    sl_up = np.zeros(2 * len(sl_ds), dtype=float)
    sh_up = np.zeros(2 * len(sh_ds), dtype=float)
    sl_up[::2] = sl_ds
    sh_up[::2] = sh_ds

    # 6) filtrar
    sl_f = np.convolve(sl_up, k1, mode="same")
    sh_f = np.convolve(sh_up, k2, mode="same")

    # 7) suma final
    return sl_f + sh_f


# —————— Ejercicio 5: comparación de SNR ——————
# %%


def compute_snr(orig, recon):
    L = min(len(orig), len(recon))
    e = orig[:L] - recon[:L]
    return 10 * np.log10(np.sum(orig[:L] ** 2) / np.sum(e**2))


# 1) carga y normaliza
fs, data = wavfile.read("marlene.wav")
if data.dtype == np.int16:
    data = data.astype(float) / 32768
elif data.dtype == np.int32:
    data = data.astype(float) / 2147483648
elif data.dtype == np.uint8:
    data = (data.astype(float) - 128) / 128
else:
    data = data.astype(float)
dataRange = (data.min(), data.max())

# 2) parámetros
wlen = 120
bg = 16
b_data = 3  # bits/muestra totales para datos
alloc = [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)]

snr_sb = []
labels = []

# 3) bucle G.722 sub-banda
for bl, bh in alloc:
    code = encoderG722(data, wlen, dataRange, bl, bh, bg)
    reco = decoderG722(code, wlen, dataRange, bl, bh, bg)
    snr = compute_snr(data, reco)
    snr_sb.append(snr)
    labels.append(f"{bl}/{bh}")
    print(f"G.722 SB b_l={bl},b_h={bh} → SNR={snr:.2f} dB")

# 4) PCM adaptable 3 bits/muestra
code_pcm = encoderAPCM(data, wlen, dataRange, b_data, bg)
reco_pcm = decoderAPCM(code_pcm, wlen, dataRange, b_data, bg)
snr_pcm = compute_snr(data, reco_pcm)
print(f"PCM adaptable b={b_data} → SNR={snr_pcm:.2f} dB")

# 5) gráfica final
x = np.arange(len(alloc))
plt.figure(figsize=(8, 4))
plt.plot(x, snr_sb, "o-", label="G.722 sub-banda")
plt.hlines(snr_pcm, x[0], x[-1], "k", "--", label="PCM adaptable (b=3)")
plt.xticks(x, labels)
plt.xlabel("Bits (b_l / b_h)")
plt.ylabel("SNR (dB)")
plt.title("Ejercicio 5: SNR vs asignación de bits")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%


# %%
