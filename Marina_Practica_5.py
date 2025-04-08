# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write

# from spectrum import *
from macros import *

plt.close("all")


# Cargamos datos
fs = 8e3  # [Hz] Frecuencia muestreo
L = 1024
N = 256  # muestras vocal /e/ muestreada
e = np.loadtxt("voc_e.asc")

plt.figure(figsize=(12, 5))
plt.plot(e)
plt.grid()
plt.xlabel("Muestras ")
plt.ylabel(" Amplitud ")
plt.title("Señal en el dominio temporal")


# Representar PSD en escala logarítmica (dB) y ω ∈ [0, π] (solo calculamos rx en la )

rx_sesgada = autocorr_sesgada(e)
PSD_estimada_sesgada = np.abs(np.fft.fft(rx_sesgada, L))
# Cogemos la mitad positiva de la autocorrelacion
PSD_estimada_sesgada = PSD_estimada_sesgada[
    : int(L / 2)
]  # Muy importante poner el int si no no se grafica


# Convertimos a escala dB
PSD_estimada_sesgada_db = 10 * np.log10(PSD_estimada_sesgada)


# %%
# Eje de frecuencias en [0, π] radianes/muestra

w = np.linspace(0, np.pi, 512)

# Graficar
plt.figure(figsize=(12, 5))
plt.plot(rx_sesgada, label="Estimada Sesgada ", linestyle="--")
plt.xlabel("Muestras ")
plt.ylabel(" Amplitud ")
plt.title("Autocorrelación Estimada Sesgada ")
plt.grid(True)
plt.legend()

# Graficar
plt.figure(figsize=(12, 5))
plt.plot(w, PSD_estimada_sesgada_db, label="Estimada Sesgada (dB)", linestyle="--")
plt.xlabel("Frecuencia (rad)")
plt.ylabel("PSD (dB)")
plt.title("PSD Estimada Sesgada (escala dB)")
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()

# %%
# Modelo LPC AR(12)

p = 12  # Orden modelo (numero de polos)
A_estimados, sigma2_estimado = coeficientes_AR_yw(e, p)


"""
# Generamos señal e sintetica
periodo=np.zeros(50); periodo[0]=1.;
excit=periodo
for nrep in range (100):
    excit=np.concatenate((excit,periodo))
e_sint = signal.lfilter([sigma2_estimado],A_estimados,excit)
rate = 8000
write('e_sint.wav',rate,e_sint.astype(np.int16))

"""


PSD_LPC = (
    sigma2_estimado / (np.abs(np.fft.fft(A_estimados, L))) ** 2
)  # C alculado con los parámetros estimados
PSD_LPC = PSD_LPC[: int(L / 2)]
PSD_LPC_db = 10 * np.log10(PSD_LPC)


# Método covarianza


A_estimados_covar, Emin = coeficientes_AR_cov(e, p)

b0 = np.sqrt(Emin)
numerador = np.abs(b0) ** 2 / (N - p)  # Factor escalado

denominador = np.abs(np.fft.fft(A_estimados_covar, L)) ** 2
PSD_covar = numerador / denominador
PSD_covar = PSD_covar[: int(L / 2)]
PSD_covar_db = 10 * np.log10(PSD_covar)


# Visualizar espectro LPC

plt.figure(figsize=(12, 5))
# Estimada sesgada - azul suave con línea discontinua
plt.plot(
    w,
    PSD_estimada_sesgada_db,
    label="Estimada Sesgada (dB)",
    linestyle="--",
    color="#6baed6",
)  # azul pastel

# LPC por autocorrelación - verde suave con línea sólida
plt.plot(
    w, PSD_LPC_db, label="LPC (autocorrelación)", linestyle="-", color="#74c476"
)  # verde pastel

# LPC por covarianza - rojo suave con línea punteada
plt.plot(
    w, PSD_covar_db, label="LPC (covarianza)", linestyle="-.", color="#fb6a4a"
)  # rojo salmón suave

# Etiquetas y estilo
plt.xlabel("Frecuencia (rad)")
plt.ylabel("PSD (dB)")
plt.title("Comparación de espectros PSD (escala dB)")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="best")

plt.tight_layout()
plt.show()


# %%
"""
# Generamos señal e sintetica
periodo=np.zeros(50); periodo[0]=1.;
excit=periodo
for nrep in range (100):
    excit=np.concatenate((excit,periodo))
    e_sint_covar=signal.lfilter(numerador,A_estimados_covar,excit)  
rate = 8000
write('e_sint_covar.wav',rate,e_sint_covar.astype(np.int16))

"""

# Análisis Homomorfico Cepstrum

# Cepstrum FFT - Periodograma
c_e = np.fft.ifft(
    PSD_estimada_sesgada_db, L
)  # Cepstrum Aplicamos logartimo para descomponer el espectro en dos sumandos.
c_e = c_e[
    1:101
]  # Exlcuimos el valor c(0) por su gran rango dinamico y tomamos los primeros 100 valores para una mejor visualizacion.

# Cepstrum LPC - Spectrum
c_e_LPC = np.fft.ifft(PSD_LPC_db, L)  # Cepstrum LPC
c_e_LPC = c_e_LPC[1:101]

plt.figure()
plt.title('Cepstrum vocal "e"')
plt.plot(c_e, label="Cepstrum FFT", color="#6baed6")
plt.plot(c_e_LPC, label="Cepstrum LPC", color="#74c476")
plt.ylabel("$C_e(n)$")
plt.xlabel("Cuefrencia (n)")
plt.grid()
plt.legend()

vocal_desconocida = np.loadtxt("voc_x.asc")

# Ejemplo de uso
vocal_identificada = identificar_vocal(vocal_desconocida, p)
print("La vocal es:", vocal_identificada)


"""
sale la "a"

"""
