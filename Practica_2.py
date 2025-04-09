#importar modulos completos
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#importar comandos especificos de algebra lineal
from scipy.linalg import inv #comando de inversion de matrices
from scipy.linalg import toeplitz #comando de generacion matrices Toeplitz




def autocorr_sesgada(x):
    N = len(x)
    r_x = np.zeros(N)  # solo k >= 0

    for k in range(N):
        suma = 0
        for n in range(k, N):
            suma += x[n] * np.conjugate(x[n - k])
        r_x[k] = suma / N  # estimación sesgada
        
    # Añadimos parte negativa (simétrica)
    r_x_neg = np.flip(r_x[1:])  # para k < 0
    r_x_total = np.concatenate([r_x_neg, r_x])  # autocorrelación completa

    return r_x_total

#np.fft.fft
"""
Sguientes cuestiones:
1. Obtener y dibujar la autocorrelacion rx(k) (k = −(N −1),...,N −1; N = 32) y el espectro Px(ω) originales del proceso AR(2). La autocorrelacion deber ´ a deducirse de las ecuaciones de Yule-Walker, y el ´
espectro a partir de la expresion analıtica correspondiente:
"""

a1 = 0
a2 = 0.81

b0 = 1
N = 32; #estimas
nu = 0;
nu_1 = 1;
sigma = 1;
k = np.arange(-N+1, N)
x = nu + np.random.randn(N)*sigma; #generar variable aleatoria
var = 1
w = np.arange(0, np.pi*2, 0.01)


def PSD(w,a1,a2,var):
  
    pds = var/np.abs(1 + a1*np.power(np.e, -1j*w) + a2*np.power(np.e, -1j*w*2))**2

    return pds


psd = PSD(w,a1,a2,var);
plt.figure()
plt.plot(w, psd)
plt.show()




# Sistema de Yule-Walker (también se podría resolver a mano)
R = np.array([[1, a1, a2],
              [a1, 1, a1],
              [a2, a1, 1]])

b = np.array([sigma, 0, 0])

r = np.linalg.solve(R, b)  # r[0] = r_x(0), r[1] = r_x(1), r[2] = r_x(2)

# Mostrar resultados
print("Autocorrelaciones teóricas:")
print(f"r_x(0) = {r[0]}")
print(f"r_x(1) = {r[1]}")
print(f"r_x(2) = {r[2]}")

"""
En lo sucesivo trabajaremos sobre una realizacion (se ´ nal) del proceso AR(2) definido anteriormente ˜
con una duracion´ N = 32. Para ello, generaremos 1000 muestras del proceso AR(2) filtrando (comando
filter en MatLab, o lfilter de scipy.signal en Python) ruido blanco Gaussiano, y nos quedaremos con la
senal de salida correspondiente a las ˜ N = 32 ultimas muestras (para evitar el periodo transitorio inicial ´
en el filtrado)
"""
N = 1000            # Número de muestras
a1 = 0              # Coeficiente AR(1)
a2 = 0.81           # Coeficiente AR(2)
sigma_v = 1         # Varianza del ruido blanco
v = np.random.randn(N) * np.sqrt(sigma_v)  # Ruido blanco

# 📌 Inicializar la señal
x = np.zeros(N)

# 📌 Generar el proceso AR(2)
for n in range(2, N):
    x[n] = -a1 * x[n-1] - a2 * x[n-2] + v[n]

