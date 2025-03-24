import numpy as np
import matplotlib.pyplot as plt

# EJERCICIO 2

N = 1000  
mu = 3.0
sigma = 1.0
                
x1 = np.random.normal(mu, sigma, N)     # Ruido blanco Gaussiano de media mu y varianza sigma^2         
x2 = np.random.randn(N)                 # Ruido blanco Gaussiano de media 0 y varianza 1
x3 = mu + sigma * np.random.randn(N)    # Ruido blanco Gaussiano de media mu y varianza sigma^2
x4 = np.random.uniform(-1, 1, N)        # Ruido blanco uniforme entre -1 y 1

seleccion = x2          # Seleccionar aquí la variable que se quiere analizar
print("Primeras 10 muestras del ruido generado:")
print(seleccion[:10])   # o seleccion[0:10] para Python 2
print("\nMedia aproximada:", np.mean(seleccion))
print("Varianza aproximada:", np.var(seleccion))

#%%
#EJERCICIO 3
def autocorr_biased(x):
    N = len(x)
    r = np.zeros(2*N - 1, dtype=complex)  # k desde -(N-1) hasta (N-1)
    for k in range(-(N-1), N):
        idx = k + (N-1)
        suma = 0
        for n in range(abs(k), N):
            suma += x[n] * np.conjugate(x[n - abs(k)])
        r[idx] = suma / N
    return r

# Seleccionamos x3 como la variable x
x = x2 # Seleccionar aquí la variable que se quiere analizar. Mire el ejercicio 2 para más detalles.

# Estimación manual (sesgada)
r_manual = autocorr_biased(x)

# Usando convolve (ecuación (2))
# r_x(k) = (1/N) * x(k) * x^*(-k) Equivale a convolucionar x(n) con x^*(-n) y dividir por N
r_conv = np.convolve(x, np.conjugate(x[::-1]), mode='full') / N

# Usando correlate de numpy (sesgado)
r_corr = np.correlate(x, x, mode='full') / N # Se divide por N para que sea no sesgada

lags = np.arange(-(N-1), N) # Genera array de enteros desde -(N-1) hasta N-1 (incluyendo -(N-1) y excluyendo N)

plt.figure(figsize=(10, 6))
plt.plot(lags, r_manual.real, label='Autocorrelación manual (sesgada)')
plt.plot(lags, r_conv.real, '--', label='Convolve (sesgada)')
plt.plot(lags, r_corr.real, ':', label='Correlate (sesgada)')
plt.title('Comparación de autocorrelaciones (sesgadas)')
plt.xlabel('Desplazamiento (k)')
plt.ylabel('r_x(k)')
plt.legend()
plt.grid(True)
plt.show()

#%%
#EJERCICIO 4
def autocorr_unbiased(x):
    N = len(x)
    r = np.correlate(x, x, mode='full') / N # NO HACE FALTA IDX
    lags = np.arange(-(N-1), N)
    
    # Normalización sin sesgo
    unbiased_r = np.zeros_like(r, dtype=float)  # Inicializa array de floats
    for i, k in enumerate(lags): #recorremos todos los valores de k
        unbiased_r[i] = r[i] * N/(N - abs(k))   # Transforma a no sesgado
    return unbiased_r, lags

r_unbiased, lags = autocorr_unbiased(x)         # LLama a la función autocorr_unbiased

plt.figure(figsize=(10, 6))
plt.plot(lags, r_corr.real, label='Sesgado (Correlate / N)')
plt.plot(lags, r_unbiased, '--', label='No sesgado')
plt.title('Comparación entre estimador sesgado y no sesgado')
plt.xlabel('Desplazamiento (k)')
plt.ylabel('r_x(k)')
plt.legend()
plt.grid(True)
plt.show()
#%%
#EJERCICIO 5
# Generamos el mismo ruido pero con media 1
x_mean1 = np.random.randn(N) + 1

# Estimador sesgado
r_corr_mean1 = np.correlate(x_mean1, x_mean1, mode='full') / N

# Estimador no sesgado
r_unbiased_mean1, lags = autocorr_unbiased(x_mean1)

plt.figure(figsize=(10, 6))
plt.plot(lags, r_corr_mean1.real, label='Sesgado con media 1')
plt.plot(lags, r_unbiased_mean1, '--', label='No sesgado con media 1')
plt.title('Autocorrelación con y sin sesgo (ruido con media 1)')
plt.xlabel('Desplazamiento (k)')
plt.ylabel('r_x(k)')
plt.legend()
plt.grid(True)
plt.show()
#%%

#EJERCICIO 6

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def autocorr_estimations(x):
    N = len(x)
    
    # Estimación sesgada
    r_sesgada = np.correlate(x, x, mode='full') / N
    
    # Estimación no sesgada
    lags = np.arange(-(N-1), N)
    r_nosesgada = np.zeros_like(r_sesgada, dtype=float)
    for i, k in enumerate(lags):
        r_nosesgada[i] = r_sesgada[i] * N / (N - abs(k))  # Corrección no sesgada
    
    return r_sesgada, r_nosesgada

# Parámetros
N = 1000  # Tamaño de la señal
num_estimates_list = [100, 1000, 10000]  # Diferentes cantidades de estimaciones

# Almacenar resultados
results = []

for num_estimates in num_estimates_list:
    sesgadas = []
    no_sesgadas = []

    for _ in range(num_estimates):  # Generar múltiples realizaciones
        x_sample = np.random.randn(N) + 1  # Ruido blanco con media unidad
        
        r_sesg, r_nosesg = autocorr_estimations(x_sample)
        
        # Guardamos solo la autocorrelación en lag 0
        sesgadas.append(r_sesg[N-1])
        no_sesgadas.append(r_nosesg[N-1])

    # Cálculo de media y varianza para cada estimador
    mean_sesgadas = np.mean(sesgadas)
    var_sesgadas = np.var(sesgadas)

    mean_nosesgadas = np.mean(no_sesgadas)
    var_nosesgadas = np.var(no_sesgadas)

    results.append([num_estimates, mean_sesgadas, var_sesgadas, mean_nosesgadas, var_nosesgadas])

# Convertimos a DataFrame para visualizar
df_results = pd.DataFrame(results, columns=["Num_Estimaciones", "Media_Sesgado", "Varianza_Sesgado", "Media_NoSesgado", "Varianza_NoSesgado"])
print(df_results)

# Gráfica de media y varianza
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(df_results["Num_Estimaciones"], df_results["Media_Sesgado"], 'o-', label="Media Sesgada")
plt.plot(df_results["Num_Estimaciones"], df_results["Media_NoSesgado"], 's-', label="Media No Sesgada")
plt.axhline(y=1, color='r', linestyle='--', label="Valor Teórico (1)")
plt.xscale("log")
plt.xlabel("Número de Estimaciones")
plt.ylabel("Media de r_x(0)")
plt.title("Media de la Autocorrelación en lag 0")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df_results["Num_Estimaciones"], df_results["Varianza_Sesgado"], 'o-', label="Varianza Sesgada")
plt.plot(df_results["Num_Estimaciones"], df_results["Varianza_NoSesgado"], 's-', label="Varianza No Sesgada")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Número de Estimaciones")
plt.ylabel("Varianza de r_x(0)")
plt.title("Varianza de la Autocorrelación en lag 0")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Definir valores de N
N_values = [100, 1000, 10000]

for N in N_values:
    E_x2 = 0  # Esperanza de x^2
    E_x = 0  # Esperanza de x
    realizaciones = 0
    
    for i in range(1, N+1):  # Iterar desde 1 hasta N
        x = np.random.randn(i)  # Generar ruido blanco
        r_x = np.correlate(x, x, mode='full')  # Autocorrelación de x
        r_x0 = r_x[len(r_x)//2]  # Valor en lag 0
        
        E_x2 += r_x0 ** 2  # Sumar valores de E[x^2]
        E_x += r_x0  # Sumar valores de E[x]
        realizaciones += 1
    
    # Calcular la esperanza dividiendo por el número de realizaciones
    E_x2 /= realizaciones
    E_x = (E_x / realizaciones) ** 2  # Elevar al cuadrado después de la media
    
    # Calcular la varianza
    varianza = E_x2 - E_x
    
    # Gráfica de la varianza
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N+1), [varianza] * N, label="Varianza Muestral")
    plt.xlabel("Tamaño de la señal")
    plt.ylabel("Varianza Muestral")
    plt.title(f"Evolución de la varianza muestral para N={N}")
    plt.legend()
    plt.grid(True)
    plt.show()
    










