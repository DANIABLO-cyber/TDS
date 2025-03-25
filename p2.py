import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#Ejercicio 1 


# Parámetros del modelo AR(2)
a1 = 0.0
a2 = 0.81
sigma_v2 = 1.0  # varianza del ruido
N = 32          # número de muestras para autocorrelación

# === 1. Resolver ecuaciones de Yule-Walker ===
# Matriz del sistema para r(0), r(1), r(2)
M_yule = np.array([
    [1,      0,     a2],         # k = 0
    [1, 1 + a2,     0],          # k = 1
    [a2,     0,     1]           # k = 2
])
b_yule = np.array([sigma_v2, 0, 0])

# Resolver el sistema lineal
r0, r1, r2 = np.dot(inv(M_yule), b_yule)

# Inicializar vector de autocorrelación rx(k) con 2N-1 muestras: k = -N+1,...,0,...,N-1
rx = np.zeros(2 * N - 1)
mid = N - 1
rx[mid] = r0
rx[mid + 1] = r1
rx[mid + 2] = r2

# === 2. Calcular autocorrelación recursiva para k > 2 ===
for i in range(mid + 2, len(rx) - 1):
    rx[i + 1] = -a1 * rx[i] - a2 * rx[i - 1]

# Simetría: r(-k) = r(k)
for i in range(mid):
    rx[i] = rx[2 * mid - i]

k_vals = np.arange(-N + 1, N)

# === 3. Calcular PSD usando FFT de la autocorrelación ===
N_fft = 1024
P_fft = np.fft.fft(rx, N_fft)
P_fft = np.abs(P_fft) ** 2
frequencies = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1))
P_fft = np.fft.fftshift(P_fft)

# === 4. Calcular PSD teórica ===
w = np.linspace(-np.pi, np.pi, N_fft, endpoint=False)
den = np.abs(1 + a1 * np.exp(-1j * w) + a2 * np.exp(-2j * w)) ** 2
P_theoretical = sigma_v2 / den

# === 5. Calcular autocorrelación estimada vía IFFT ===
rx_estimada = np.fft.ifft(P_theoretical).real
rx_estimada = np.fft.fftshift(rx_estimada)
k_est = np.arange(-N_fft // 2, N_fft // 2)

# === 6. Gráficas ===

# Autocorrelación por Yule-Walker + extensión recursiva
plt.figure()
plt.stem(k_vals, rx)
plt.title("Autocorrelación $r_x(k)$ (Yule-Walker + Recursiva)")
plt.xlabel("k")
plt.ylabel("$r_x(k)$")
plt.grid()
plt.tight_layout()
plt.show()

# Espectro obtenido por FFT de autocorrelación
plt.figure()
plt.plot(frequencies, P_fft)
plt.title("Espectro $P_x(\\omega)$ vía FFT de $r_x(k)$")
plt.xlabel("Frecuencia")
plt.ylabel("$P_x(\\omega)$")
plt.grid()
plt.tight_layout()
plt.show()

# Autocorrelación estimada desde IFFT de PSD teórica
plt.figure()
plt.stem(k_est[N_fft // 2 - 40: N_fft // 2 + 40], rx_estimada[N_fft // 2 - 40: N_fft // 2 + 40])
plt.title("Autocorrelación desde IFFT($P_x(\\omega)$)")
plt.xlabel("k")
plt.ylabel("$\\hat{r}_x(k)$")
plt.grid()
plt.tight_layout()
plt.show()

# Espectro teórico directo
plt.figure()
plt.plot(w, P_theoretical)
plt.title("Espectro teórico $P_x(\\omega)$")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("$P_x(\\omega)$")
plt.grid()
plt.tight_layout()
plt.show()

#%%

#Ejercicio 2


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Parámetros del AR(2)
a1 = 0.0
a2 = 0.81
sigma_v2 = 1.0
N_real = 32           # muestras que queremos quedarnos
N_total = 1000        # muestras totales para evitar transitorio

# === 1. Generar ruido blanco Gaussiano v(n) ===
np.random.seed(42)  # reproducibilidad
v = np.random.normal(loc=0, scale=np.sqrt(sigma_v2), size=N_total)

# === 2. Definir coeficientes del filtro AR(2) ===
# Modelo: x(n) + a1 x(n-1) + a2 x(n-2) = v(n)
# En lfilter, eso es: lfilter(b=[1], a=[1, a1, a2], input=v)
b = [1]                  # numerador
a = [1, a1, a2]          # denominador (AR)

# === 3. Filtrar para obtener x(n) ===
x = lfilter(b, a, v)

# === 4. Tomar las últimas N_real muestras para evitar el transitorio inicial ===
x_final = x[-N_real:]  # x(n) desde la muestra 968 a 999

# === 5. Representar la realización ===
plt.figure()
plt.stem(np.arange(len(x_final)), x_final)
plt.title("Realización del proceso AR(2) (últimas 32 muestras)")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid()
plt.tight_layout()
plt.show()
#%%
#Ejercicio 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# -----------------------------
# Función sesgada
def autocorr_biased(x):
    N = len(x)
    r = np.zeros(2 * N - 1, dtype=complex)  # k desde -(N-1) hasta (N-1)
    for k in range(-(N - 1), N):
        idx = k + (N - 1)
        suma = 0
        for n in range(abs(k), N):
            suma += x[n] * np.conjugate(x[n - abs(k)])
        r[idx] = suma / N
    return r.real  # asumimos señal real

# -----------------------------
# Función no sesgada
def autocorr_unbiased(x):
    N = len(x)
    r = np.correlate(x, x, mode='full') / N  # autocorrelación sesgada
    lags = np.arange(-N + 1, N)

    unbiased_r = np.zeros_like(r, dtype=float)
    for i, k in enumerate(lags):
        unbiased_r[i] = r[i] * N / (N - abs(k))
    return unbiased_r, lags

# -----------------------------
# Parámetros del proceso AR(2)
a1 = 0.0
a2 = 0.81
sigma_v2 = 1.0
N = 32
N_total = 1000

# -----------------------------
# Generar una realización del proceso AR(2)
np.random.seed(42)
v = np.random.normal(0, np.sqrt(sigma_v2), N_total)
b = [1]
a = [1, a1, a2]
x = lfilter(b, a, v)
x_real = x[-N:]

# -----------------------------
# Estimar autocorrelación con y sin sesgo usando tus funciones
r_sesgado = autocorr_biased(x_real)
r_nosesgado, lags = autocorr_unbiased(x_real)

# -----------------------------
# Autocorrelación teórica (Yule-Walker + recursiva)
M_yule = np.array([
    [1,      0,     a2],
    [1, 1 + a2,     0],
    [a2,     0,     1]
])
b_yule = np.array([sigma_v2, 0, 0])
r0, r1, r2 = np.dot(np.linalg.inv(M_yule), b_yule)

r_teorica = np.zeros(2 * N - 1)
mid = N - 1
r_teorica[mid] = r0
r_teorica[mid + 1] = r1
r_teorica[mid + 2] = r2

# extensión recursiva hacia adelante
for i in range(mid + 2, len(r_teorica) - 1):
    r_teorica[i + 1] = -a1 * r_teorica[i] - a2 * r_teorica[i - 1]

# simetría hacia atrás
for i in range(mid):
    r_teorica[i] = r_teorica[2 * mid - i]

k_teo = np.arange(-N + 1, N)

# -----------------------------
# Gráfico comparativo
plt.figure(figsize=(10, 5))
plt.plot(k_teo, r_teorica, label="Autocorrelación teórica", linewidth=2)
plt.plot(k_teo, r_sesgado, '--', label="Estimación sesgada")
plt.plot(lags, r_nosesgado, '--', label="Estimación no sesgada")
plt.title("Comparación de autocorrelaciones")
plt.xlabel("k")
plt.ylabel("$r_x(k)$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%

#Ejercicio 4
# --- FFT de autocorrelaciones estimadas para obtener PSDs ---

N_fft = 1024  # resolución espectral
frecs = np.linspace(-np.pi, np.pi, N_fft, endpoint=False)

# FFT de autocorrelación sesgada
PSD_sesgada = np.fft.fftshift(np.abs(np.fft.fft(r_sesgado, N_fft)).real)

# FFT de autocorrelación no sesgada
PSD_nosesgada = np.fft.fftshift(np.abs(np.fft.fft(r_nosesgado, N_fft)).real)

# --- Cálculo del espectro teórico para comparación ---
den = np.abs(1 + a1 * np.exp(-1j * frecs) + a2 * np.exp(-2j * frecs)) ** 2
PSD_teorica = sigma_v2 / den

# --- Graficar las 3 PSDs juntas ---
plt.figure(figsize=(10,5))
plt.plot(frecs, PSD_teorica, label="PSD teórica", linewidth=2)
plt.plot(frecs, PSD_sesgada, '--', label="PSD desde estimación sesgada")
plt.plot(frecs, PSD_nosesgada, '--', label="PSD desde estimación no sesgada")
plt.title("Comparación de densidades espectrales (PSD)")
plt.xlabel("Frecuencia (rad/muestra)")
plt.ylabel("$P_x(\\omega)$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
#Ejercicio 5
from scipy.linalg import toeplitz

# Número de muestras de x
N = len(x_real)

# Extraemos r(0), r(1), r(2) desde la autocorrelación sesgada
# Recordemos que r_sesgado tiene índices de -N+1 a N-1
mid = N - 1
r0 = r_sesgado[mid]
r1 = r_sesgado[mid + 1]
r2 = r_sesgado[mid + 2]

# Construimos la matriz de Toeplitz
R = toeplitz([r0, r1])  # fila inicial
rhs = -np.array([r1, r2])  # lado derecho del sistema

# Resolvemos el sistema
a_est = np.linalg.solve(R, rhs)

a1_est, a2_est = a_est
print("Estimación de los coeficientes AR(2):")
print(f"a1 ≈ {a1_est:.4f}")
print(f"a2 ≈ {a2_est:.4f}")

#%%

#Ejercicio 6

# === Estimar sigma_v^2 a partir de Yule-Walker ===
# Ecuación: r(0) + a1*r(1) + a2*r(2) = sigma_v^2
sigma_v2_est = r0 + a1_est * r1 + a2_est * r2

# === Calcular PSD del modelo AR(2) estimado ===
frecs = np.linspace(-np.pi, np.pi, N_fft, endpoint=False)
den_est = np.abs(1 + a1_est * np.exp(-1j * frecs) + a2_est * np.exp(-2j * frecs)) ** 2
PSD_modelo_estimado = sigma_v2_est / den_est

# === Recalcular PSD real y PSD por periodograma para comparar ===
den_real = np.abs(1 + a1 * np.exp(-1j * frecs) + a2 * np.exp(-2j * frecs)) ** 2
PSD_real = sigma_v2 / den_real
PSD_periodograma = PSD_sesgada  # ya lo tenías en apartado 4

# === Graficar comparación de las 3 PSDs ===
plt.figure(figsize=(10, 5))
plt.plot(frecs, PSD_real, label="PSD real (modelo verdadero)", linewidth=2)
plt.plot(frecs, PSD_periodograma, '--', label="Periodograma (desde r sesgada)")
plt.plot(frecs, PSD_modelo_estimado, '--', label="PSD del modelo AR(2) estimado")
plt.title("Comparación de PSDs")
plt.xlabel("Frecuencia [rad/muestra]")
plt.ylabel("$P_x(\\omega)$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

