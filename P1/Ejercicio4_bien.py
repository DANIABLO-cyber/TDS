import numpy as np
from scipy.io import wavfile
import sounddevice as sd

# Cargar archivo de audio
fs, x = wavfile.read("altura.wav")
x = x.astype(np.float32)
x = x / np.max(np.abs(x))  # Normalización [-1, 1]

def cuantificar(x, bits, usar_dither=False):
    niveles = 2 ** bits
    delta = 2 / niveles
    if usar_dither:
        dither = np.random.triangular(-delta/2, 0, delta/2, size=x.shape)
        x = x + dither
    xq = np.round(x / delta) * delta
    return np.clip(xq, -1, 1)

# Cuantificación con 3 bits
xq3 = cuantificar(x, 3, usar_dither=False)
xq3_d = cuantificar(x, 3, usar_dither=True)

# Cuantificación con 5 bits
xq5 = cuantificar(x, 5, usar_dither=False)
xq5_d = cuantificar(x, 5, usar_dither=True)

# Reproducción una por una
print("Original")
sd.play(x, fs); sd.wait()

print("3 bits sin dither")
sd.play(xq3, fs); sd.wait()

print("3 bits con dither")
sd.play(xq3_d, fs); sd.wait()

print("5 bits sin dither")
sd.play(xq5, fs); sd.wait()

print("5 bits con dither")
sd.play(xq5_d, fs); sd.wait()

#%%

def calcular_snr(x, xq):
    error = x - xq
    ps = np.mean(x ** 2)
    pe = np.mean(error ** 2)
    return 10 * np.log10(ps / pe)

print("SNR 3 bits sin dither:", calcular_snr(x, xq3))
print("SNR 3 bits con dither:", calcular_snr(x, xq3_d))
print("SNR 5 bits sin dither:", calcular_snr(x, xq5))
print("SNR 5 bits con dither:", calcular_snr(x, xq5_d))