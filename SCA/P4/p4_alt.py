# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, convolve
from scipy.io import wavfile

from scalib import (
    UniformSQ,
    FixedLengthCoder,
    snr as calc_snr,
    signalRange,
    getG722lpf,
)
# %%


# --- APCM (Adaptive PCM) ---


def apcm_coder(x, win, rng, nbits, nbits_g):
    output = []
    max_val = rng[1]
    for i in range(0, len(x), win):
        chunk = x[i : i + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        g = np.max(np.abs(chunk))
        if g == 0:
            g = 1.0
        norm_chunk = chunk / g
        idx_data = UniformSQ(nbits, (-1, 1)).encode(norm_chunk)
        idx_gain = UniformSQ(nbits_g, (0, max_val)).encode([g])
        bits_gain = FixedLengthCoder(nbits_g).encode(idx_gain)
        bits_data = FixedLengthCoder(nbits).encode(idx_data)
        output.extend(bits_gain)
        output.extend(bits_data)
    return np.array(output, dtype=np.uint8)


def apcm_decoder(bitstream, win, rng, nbits, nbits_g):
    if nbits == 0:
        return np.zeros(0, dtype=float)
    result = []
    max_val = rng[1]
    block_size = nbits_g + nbits * win
    for j in range(0, len(bitstream), block_size):
        seg = bitstream[j : j + block_size]
        bits_g = seg[:nbits_g]
        bits_d = seg[nbits_g:]
        idx_g = FixedLengthCoder(nbits_g).decode(bits_g)
        idx_d = FixedLengthCoder(nbits).decode(bits_d)
        g = UniformSQ(nbits_g, (0, max_val)).decode(idx_g)[0]
        norm = UniformSQ(nbits, (-1, 1)).decode(idx_d)
        result.extend(norm * g)
    return np.array(result, dtype=float)


def encode_pcm(signal, value_range, nbits):
    quant = UniformSQ(nbits, value_range)
    idx = quant.encode(signal)
    return FixedLengthCoder(nbits).encode(idx)


def decode_pcm(bitstream, value_range, nbits):
    idx = FixedLengthCoder(nbits).decode(bitstream)
    quant = UniformSQ(nbits, value_range)
    return quant.decode(idx)


# %%
# --- Ejercicio 1: PCM vs APCM ---
fs, audio_raw = wavfile.read("marlene.wav")
audio = audio_raw.astype(float)
rng = signalRange(audio_raw)

window_size = 120
gain_bits = 16
bit_depths = np.arange(1, 9)
snr_apcm = []
snr_pcm = []

for bits in bit_depths:
    print(f"Procesando {bits} bits...")
    # APCM
    apcm_encoded = apcm_coder(audio, window_size, rng, bits, gain_bits)
    apcm_decoded = apcm_decoder(apcm_encoded, window_size, rng, bits, gain_bits)
    snr_apcm.append(calc_snr(audio, apcm_decoded))
    # PCM
    pcm_encoded = encode_pcm(audio, rng, bits)
    pcm_decoded = decode_pcm(pcm_encoded, rng, bits)
    snr_pcm.append(calc_snr(audio, pcm_decoded))

plt.figure(figsize=(10, 5))
plt.plot(bit_depths, snr_pcm, "o--", label="PCM", color="red")
plt.plot(bit_depths, snr_apcm, "s-", label="APCM", color="blue")
plt.xlabel("Bits por muestra")
plt.ylabel("SNR (dB)")
plt.title("Comparación SNR: PCM vs APCM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
# --- Ejercicio 2: Filtros QMF ---


def qmf_filters(ntaps, cutoff):
    h1 = firwin(ntaps, cutoff, window="hamming")
    n = np.arange(ntaps)
    h2 = ((-1) ** n) * h1
    k1 = 2 * h1
    k2 = -2 * h2
    return h1, h2, k1, k2


ntaps = 24
cut = 0.5
h1, h2, k1, k2 = qmf_filters(ntaps, cut)

print("h1:", h1)
print("h2:", h2)
print("k1:", k1)
print("k2:", k2)

for label, coeffs in {"h1": h1, "h2": h2, "k1": k1, "k2": k2}.items():
    w, H = freqz(coeffs, worN=1024)
    plt.figure()
    plt.title(f"Respuesta en frecuencia de {label}")
    plt.plot(w / np.pi, 20 * np.log10(np.abs(H)))
    plt.xlabel("Frecuencia normalizada (×π rad/muestra)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True)
plt.show()
# %%
# --- Ejercicio 3: Respuesta global ---


def resp_total(h1, h2, k1, k2):
    return 0.5 * (convolve(h1, k1) + convolve(h2, k2))


def plot_resp(g, label):
    w, H = freqz(g, worN=2048)
    plt.plot(w / np.pi, 20 * np.log10(np.abs(H)), label=label)


def caso_y_grafica(label, h1):
    n = np.arange(len(h1))
    h2 = ((-1) ** n) * h1
    k1 = 2 * h1
    k2 = -2 * h2
    g = resp_total(h1, h2, k1, k2)
    delay = (len(g) - 1) / 2
    print(f"{label}: longitud g = {len(g)}, retardo = {delay:.1f} muestras")
    plot_resp(g, label)


plt.figure(figsize=(8, 5))
h1_orig, _, _, _ = qmf_filters(ntaps, cut)
caso_y_grafica("Cutoff 0.50xNyquist", h1_orig)
cut2 = cut + 0.025
h1_high, _, _, _ = qmf_filters(ntaps, cut2)
caso_y_grafica(f"Cutoff {cut2:.3f}xNyquist", h1_high)


plt.title("Respuesta en frecuencia del sistema completo g(n)")
plt.xlabel("Frecuencia normalizada (×π rad/muestra)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.show()
# %%
# --- Ejercicio 4: G.722 encoder/decoder  ---


def getBitsPerChannel(nBits, wlen, bLB, bHB, bG):
    bGLB = bG if bLB > 0 else 0
    bGHB = bG if bHB > 0 else 0
    wSize = wlen * (bLB + bHB) + bGLB + bGHB

    if wSize == 0:
        return 0, 0
    nWindows = nBits // wSize
    nSamples = nWindows * wlen
    # Ajuste para bits restantes (parcial ventana)
    remBits = nBits % wSize

    if (bLB + bHB) > 0:
        nSamples += max(remBits - bGLB - bGHB, 0) // (bLB + bHB)
    nLB = int(nSamples * bLB + np.ceil(nSamples / wlen) * bGLB)
    nHB = nBits - nLB

    return nLB, nHB


def g722_coder(x, win, rng, nbits_lo, nbits_hi, nbits_g):
    # Obtención de filtros QMF
    lpf = getG722lpf()
    n = np.arange(len(lpf))
    h_lo = lpf.copy()
    h_hi = np.power(-1, n) * lpf

    # Filtrado de la señal original
    y_lo = np.convolve(x, h_lo, mode="full")
    y_hi = np.convolve(x, h_hi, mode="full")

    # Submuestreo por 2
    y_lo_ds = y_lo[::2]
    y_hi_ds = y_hi[::2]

    # Codificación APCM independiente por banda
    bits_g_lo = nbits_g if nbits_lo > 0 else 0
    bits_g_hi = nbits_g if nbits_hi > 0 else 0
    coded_lo = apcm_coder(y_lo_ds, win, rng, nbits_lo, bits_g_lo)
    coded_hi = apcm_coder(y_hi_ds, win, rng, nbits_hi, bits_g_hi)

    # Unir los streams de ambas bandas
    return np.concatenate((coded_lo, coded_hi))


def g722_decoder(bitstream, win, rng, nbits_lo, nbits_hi, nbits_g):
    # Filtros de síntesis QMF
    lpf = getG722lpf()
    n = np.arange(len(lpf))
    k_lo = 2 * lpf
    k_hi = (-1) ** (n + 1) * k_lo

    # Determinar la cantidad de bits para cada banda
    bits_lo_total, bits_hi_total = getBitsPerChannel(
        len(bitstream), win, nbits_lo, nbits_hi, nbits_g
    )
    stream_lo = bitstream[:bits_lo_total]
    stream_hi = bitstream[bits_lo_total : bits_lo_total + bits_hi_total]

    # Decodificación APCM por banda
    bits_g_lo = nbits_g if nbits_lo > 0 else 0
    bits_g_hi = nbits_g if nbits_hi > 0 else 0
    rec_lo = apcm_decoder(stream_lo, win, rng, nbits_lo, bits_g_lo)
    rec_hi = apcm_decoder(stream_hi, win, rng, nbits_hi, bits_g_hi)

    # Upsample y filtrado de síntesis
    rec_lo_up = np.zeros(len(rec_lo) * 2)
    rec_lo_up[::2] = rec_lo
    rec_hi_up = np.zeros(len(rec_hi) * 2)
    rec_hi_up[::2] = rec_hi

    out_lo = (
        np.convolve(rec_lo_up, k_lo, mode="full") if rec_lo_up.size > 0 else np.zeros(0)
    )
    out_hi = (
        np.convolve(rec_hi_up, k_hi, mode="full") if rec_hi_up.size > 0 else np.zeros(0)
    )

    # Suma de ambas bandas, igualando longitud
    L = max(len(out_lo), len(out_hi))
    out_lo = np.pad(out_lo, (0, L - len(out_lo)))
    out_hi = np.pad(out_hi, (0, L - len(out_hi)))
    return out_lo + out_hi


# %%
# --- Ejercicio 5: Comparativa G.722 vs APCM ---


def group_delay_v2(h):
    n = np.arange(len(h))
    e = h * h
    return np.dot(n, e) / np.sum(e)


def comparar_g722_apcm(audio, bits_total, win_size, bits_gain):
    configs = [(blo, bits_total - blo) for blo in range(bits_total + 1)]
    h1 = getG722lpf()
    n = np.arange(len(h1))
    h2 = (-1) ** n * h1
    k1 = 2 * h1
    k2 = -2 * h2
    g722_resp = 0.5 * (convolve(h1, k1) + convolve(h2, k2))
    delay = int(round(group_delay_v2(g722_resp)))
    snr_g722, snr_apcm = [], []
    for blo, bhi in configs:
        print(f"Probando configuración: b_lo={blo}, b_hi={bhi}")
        coded_g722 = g722_coder(
            audio, win_size, signalRange(audio), blo, bhi, bits_gain
        )
        decoded_g722 = g722_decoder(
            coded_g722, win_size, signalRange(audio), blo, bhi, bits_gain
        )
        rec_g722 = decoded_g722[delay:]
        if len(rec_g722) < len(audio):
            rec_g722 = np.pad(rec_g722, (0, len(audio) - len(rec_g722)))
        else:
            rec_g722 = rec_g722[: len(audio)]
        snr_g722.append(calc_snr(audio, rec_g722))
        bits_apcm = bits_total // 2
        coded_apcm = apcm_coder(
            audio, win_size, signalRange(audio), bits_apcm, bits_gain
        )
        decoded_apcm = apcm_decoder(
            coded_apcm, win_size, signalRange(audio), bits_apcm, bits_gain
        )
        L = min(len(audio), len(decoded_apcm))
        snr_apcm.append(calc_snr(audio[:L], decoded_apcm[:L]))
    return configs, snr_g722, snr_apcm


fs, audio_raw = wavfile.read("marlene.wav")
audio_norm = audio_raw.astype(float) / np.max(np.abs(audio_raw))
bits_total = 6
win_size = 120
bits_gain = 16

configs, snr_g722, snr_apcm = comparar_g722_apcm(
    audio_norm, bits_total, win_size, bits_gain
)
labels = [f"({blo},{bhi})" for blo, bhi in configs]

plt.figure(figsize=(10, 5))
plt.plot(range(len(configs)), snr_apcm, "^--", color="blue", label="APCM (b=3)")
plt.plot(range(len(configs)), snr_g722, "o-", color="red", label="G.722")
plt.xticks(range(len(configs)), labels)
plt.xlabel("Distribución de bits (b_lo, b_hi)")
plt.ylabel("SNR (dB)")
plt.title("SNR: G.722 vs APCM (3 bits/muestra)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

max_snr = max(snr_g722)
idx_max = snr_g722.index(max_snr)
best_blo, best_bhi = configs[idx_max]
print(f"SNR máxima G.722: {max_snr:.2f} dB con (b_lo, b_hi)=({best_blo},{best_bhi})")
# %%
