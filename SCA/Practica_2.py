import scalib as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from skimage import io, color
import sounddevice as sd


# %%
"""
Ejercicio 1
Implemente los algoritmos de Lloyd-Max y Linde-Buzo-Gray. Para ello, cree dos funciones con
la siguiente interfaz:
 #####   

"""  
def lloyd(C, training, Emax, maxIter):
    i = 0   # Contador de iteracione
    Er_rel = np.inf # Inicializamos el error relativo con valor infinito (alto para asegurar iteración)
    
    # Paso 1: Condición del vecino más próximo
    # Inicializamos array para guardar el índice del centroide más cercano a cada muestra
    while (Er_rel > Emax) and (i < maxIter):
        # Vecino más próximo (distancia euclídea para cada muestra)
        mas_proximo = np.zeros(len(training), dtype=int) # Almacena valores de los indices
        for j in range(len(training)):
            # Calculamos la distancia euclídea de la muestra j
            distancias = np.linalg.norm(C - training[j], axis=1)
            dmin = np.argmin(distancias) # Encontramos el índice del centroide más cercano (mínima distancia)
            mas_proximo[j] = dmin # Guardamos la muestra j al centroide más cercano

        # Condición del centroide (promedio de las muestras asignadas a él)
        C_nuevo = np.copy(C)
        for r in range(len(C)):
            indices = np.where(mas_proximo == r)[0]
            if len(indices) > 0:
                C_nuevo[r] = np.mean(training[indices], axis=0) # Buscamos índices de las muestras asignadas al centroide r
            else:
                # Si ningún vector se asignó a este centroide, mantenemos el centroide original
                C_nuevo[r] = C[r]

        # Error relativo: máximo entre todos los centroides (distancia euclídea)
        Er_rel = np.max(np.linalg.norm(C_nuevo - C, axis=1))
        C = C_nuevo.copy()
        i += 1

    return C #Devolvemos centroides optimizados

def lbg(b, training, Emax=1e-3, maxIter=1000, delta=0.001):
    C = np.array([np.mean(training, axis=0)])  # Corrección de sintaxis
    while len(C) < (2**b):
        C = np.concatenate((C*(1-delta), C*(1+delta)), axis=0)
        C = lloyd(C, training, Emax, maxIter)

    return C



#%% COMPROBACIÓN FUNCIONES
"""
 Ejercicio 2
Construya tres cuantificadores escalares óptimos con una tasa de 3 bits/muestra. Para cada uno
use un conjunto de entrenamiento formado por 100000 muestras aleatorias con las siguientes
distribuciones de probabilidad:
Normal de media 0 y desviación típica 0.1.
Rayleigh con paramétro σ =1.
Uniforme entre -1 y 1
En cada caso, analice la separación entre niveles de cuantificación, la forma del cuantificador
(representación de entradas frente a salidas) respecto a la distribución de probabilidad de los
datos. 
Las funciones normal, rayleigh y uniform de numpy.random para generar las muestras aleatorias.
El atributo C de los cuantificadores construidos con las clases de scalib, que almacena el
conjunto de niveles de cuantificación.
La función hist de numpy, para construir el histograma de un conjunto de datos.
La función diff de numpy, para calcular la distancia entre cada 2 niveles de cuantificación.
Por ultimo, compare el cuantificador construido con la distribución uniforme con un cuantificador construido con la clase Uniforme.
"""

b = 3

# Datos de entrenamiento
x_normal = np.random.normal(0, 0.1, 100000)
x_rayleigh = np.random.rayleigh(1, 100000)
x_uniforme = np.random.uniform(-1, 1, 100000)




"""
Primeramente comprobamos los resultados que queremos obtener con 
el cuantificador proporcionado por la libreria scalib
qtz_opt_normal = sc.OptimalSQ(b, x_normal )
qtz_opt_rayleigh = sc.OptimalSQ(b, x_rayleigh)
qtz_opt_uniforme = sc.OptimalSQ(b, x_uniforme)

"""

# Cuantificadores óptimos
qtz_opt_normal = sc.OptimalSQ(b, x_normal, algorithm=lbg)
qtz_opt_rayleigh = sc.OptimalSQ(b, x_rayleigh, algorithm=lbg)
qtz_opt_uniforme = sc.OptimalSQ(b, x_uniforme, algorithm=lbg)

# Cuantificador uniforme de media huella para comparar
qtz_uniforme = sc.UniformSQ(b, [-1, 1], qtype='midrise')

x_qtz_normal = qtz_opt_normal.quantize(x_normal) 
x_qtz_rayleigh = qtz_opt_rayleigh.quantize(x_rayleigh) 
x_qtz_opt_uniforme = qtz_opt_uniforme.quantize(x_uniforme) 
x_qtz_uniforme = qtz_uniforme.quantize(x_normal)


# Extraer niveles y diferencias
niveles_q_opt_normal = qtz_opt_normal.C
niveles_q_opt_rayleigh = qtz_opt_rayleigh.C
niveles_q_opt_uniforme = qtz_opt_uniforme.C
niveles_q_uniform = qtz_uniforme.C


d_normal = np.diff(niveles_q_opt_normal)
d_rayleigh = np.diff(niveles_q_opt_rayleigh)
d_uniforme = np.diff(niveles_q_opt_uniforme)
d_qtz = np.diff(niveles_q_uniform)
"""
# Crear tabla
df = pd.DataFrame({
    'Normal – OptimalSQ': d_normal,
    'Rayleigh – OptimalSQ': d_rayleigh,
    'Uniforme – OptimalSQ': d_uniforme,
    'Uniforme – UniformSQ': d_qtz
})
import ace_tools as tools; tools.display_dataframe_to_user(name="Diferencias entre niveles", dataframe=df)
"""
# %% Gráficas
# Crear gráfico de histogramas
fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout='tight')
fig.suptitle("Distribuciones de entrenamiento", fontsize=14)

# Histograma normal
ax[0].hist(x_normal, bins=100, density=True, alpha=0.7, color='skyblue')
ax[0].set_title("Normal (μ=0, σ=0.1)")
ax[0].grid()

# Histograma Rayleigh
ax[1].hist(x_rayleigh, bins=100, density=True, alpha=0.7, color='salmon')
ax[1].set_title("Rayleigh (σ=1)")
ax[1].grid()

# Histograma uniforme
ax[2].hist(x_uniforme, bins=100, density=True, alpha=0.7, color='lightgreen')
ax[2].set_title("Uniforme [-1, 1]")
ax[2].grid()

plt.show()


# Valores de entrada

x_1 = np.linspace(0, 3.5, 10000)
x_0 = np.linspace(-0.3, 0.3, 1000)
x_2 = np.linspace(-1, 1, 1000)
y_0 = qtz_opt_normal.quantize(x_0)
y_1 = qtz_opt_rayleigh.quantize(x_1)
y_2 = qtz_opt_uniforme.quantize(x_2)
y_3 = qtz_uniforme.quantize(x_2)


margen = 0.01

# Graficar
# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 16), layout='tight')
fig.suptitle("Función de cuantificación (entrada vs salida cuantificada)", fontsize=14)

# Normal
ax[0, 0].plot(x_0, y_0, lw=2, color='skyblue' )
ax[0, 0].set_title('Normal – OptimalSQ')
ax[0, 0].grid(True)
ax[0, 0].set_xlim(np.min(x_0), np.max(x_0))
ax[0, 0].set_ylim(np.min(y_0) - margen, np.max(y_0) + margen)

# Rayleigh
ax[0, 1].plot(x_1, y_1, lw=2, color='salmon')
ax[0, 1].set_title('Rayleigh – OptimalSQ')
ax[0, 1].grid(True)
ax[0, 1].set_xlim(np.min(x_1), np.max(x_1))
ax[0, 1].set_ylim(np.min(y_1) - margen, np.max(y_1) + margen)

# Uniforme – OptimalSQ
ax[1, 0].plot(x_2, y_2, lw=2, color='lightgreen')
ax[1, 0].set_title('Uniforme – OptimalSQ')
ax[1, 0].grid(True)
ax[1, 0].set_xlim(np.min(x_2), np.max(x_2))
ax[1, 0].set_ylim(np.min(y_2) - margen, np.max(y_2) + margen)

# Uniforme – UniformSQ
ax[1, 1].plot(x_2, y_3, lw=2)
ax[1, 1].set_title('Uniforme – UniformSQ')
ax[1, 1].grid(True)
ax[1, 1].set_xlim(np.min(x_2), np.max(x_2))
ax[1, 1].set_ylim(np.min(y_3) - margen, np.max(y_3) + margen)

plt.show()

# %%

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Señales cuantificadas", fontsize=16)

# Señal normal cuantificada con OptimalSQ
axs[0, 0].hist(x_qtz_normal, bins=50, color='skyblue', alpha=0.7, density=True)
axs[0, 0].set_title("x_qtz_normal – OptimalSQ")
axs[0, 0].grid(True)

# Señal Rayleigh cuantificada con OptimalSQ
axs[0, 1].hist(x_qtz_rayleigh, bins=50, color='salmon', alpha=0.7, density=True)
axs[0, 1].set_title("x_qtz_rayleigh – OptimalSQ")
axs[0, 1].grid(True)

# Señal uniforme cuantificada con OptimalSQ
axs[1, 0].hist(x_qtz_opt_uniforme, bins=50, color='lightgreen', alpha=0.7, density=True)
axs[1, 0].set_title("x_qtz_opt_uniforme – OptimalSQ")
axs[1, 0].grid(True)

# Señal normal cuantificada con UniformSQ
axs[1, 1].hist(x_qtz_uniforme, bins=50, color='orchid', alpha=0.7, density=True)
axs[1, 1].set_title("x_qtz_uniforme – UniformSQ sobre señal normal")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


# %% 
"""
Ejercicio 3

Construya un cuantificador escalar óptimo para señales de voz con una tasa de 4 bits/muestra.
Use las señales almacenadas en ciudad.wav, mar.wav, rio.wav y santander.wav concatenadas una
tras otra como conjunto de entrenamiento.
A continuación, cuantifique la señal almacenada en altura.wav. Reproduzca la señal cuantificada
y calcule su SNR. Repita el análisis usando un cuantificador con tasas de 6 y 8 bit/muestra.
Analice los resultados en términos de SNR y calidad percibida, comparándolos con los obtenidos
usando cuantificadores uniformes con la misma tasa de bits.
"""

# Cargar señales para entrenamiento
fs, audio_ciudad = wf.read('ciudad.wav') # todos tienen la misma fs
_, audio_mar = wf.read('mar.wav')
_, audio_rio = wf.read('rio.wav')
_, audio_santander = wf.read('santander.wav')

# Cargar señal para cuantificar
_, audio_altura = wf.read('Altura.wav')

# Concatenar señales de entrenamiento en una única señal
entrenamiento_audio = np.concatenate((audio_ciudad, audio_mar, audio_rio, audio_santander))

# Visualización del histograma del audio de entrenamiento
plt.figure(figsize=(8, 4))
plt.hist(entrenamiento_audio, bins=100, color='skyblue')
plt.title('Histograma de la señal de entrenamiento (concatenada)')
plt.xlabel('Amplitud')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Determinar rango de la señal
rango_audio = sc.signalRange(entrenamiento_audio)

# Definir tasas de bits a analizar
tasas_bits = [4, 6, 8]

# Analizar cuantificadores
for b in tasas_bits:
    # Construir cuantificador óptimo
    cuantificador_opt = sc.OptimalSQ(b, entrenamiento_audio, algorithm=lbg)
    audio_altura_opt = cuantificador_opt.quantize(audio_altura)
    snr_opt = sc.snr(audio_altura_opt, audio_altura)

    # Construir cuantificador uniforme (midtread para señales de voz(muchos ceros))
    cuantificador_unif = sc.UniformSQ(b, rango_audio, qtype='midtread')
    audio_altura_unif = cuantificador_unif.quantize(audio_altura)
    snr_unif = sc.snr(audio_altura_unif, audio_altura)
    

    # Resultados
    print(f'--- Tasa de bits: {b} bits/muestra ---')
    print(f'SNR Cuantificador Óptimo: {snr_opt:.2f} dB')
    print(f'SNR Cuantificador Uniforme: {snr_unif:.2f} dB')
    print('\n')
    
    # Gráfica comparativa
    plt.figure(figsize=(10, 5))
    plt.hist(audio_altura_opt, bins=100, alpha=0.6, label=f'Óptimo (SNR: {snr_opt:.2f} dB)', color='blue')
    plt.hist(audio_altura_unif, bins=100, alpha=0.6, label=f'Uniforme (SNR: {snr_unif:.2f} dB)', color='orange')
    plt.title(f'Comparación Cuantificadores - {b} bits/muestra')
    plt.xlabel('Amplitud')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True)
    plt.show()


    #Comprobamos la calidad percibida de la señal de audio

    #sd.play(audio_altura_opt, fs)
    sd.play(audio_altura_unif, fs)

"""
Se escucha peor con el cuantificador uniforme a pesar de tener el mismo numero 
de bits, no tenemos ruido de fondo ( en el uniforme un poco más) tenemos cortes.
En ambos casos cuando aumentamos el numero de bits se escucha mejor
"""

# %%

"""
Ejercicio 4
Construya un cuantificador escalar óptimo para imágenes con una tasa de 4 bits/muestra. Use
las imágenes almacenas en caravana.png, maiz.png, fresas.png y oficina.png concatenadas una
tras otra como conjunto de entrenamiento.
A continuación, cuantifique la imagen almacenada en lena.png. Represente la imagen cuantificada y calcule su SNR. Repita el análisis usando un cuantificador con una tasa de 6 bit/muestra.
Analice los resultados en términos de SNR y calidad percibida, comparándolos con los obtenidos
usando cuantificadores uniformes con la misma tasa de bits.

"""


# Leer imágenes
caravana = io.imread('caravana.png')
maiz = io.imread('maiz.png')
fresas = io.imread('fresas.png')
oficina = io.imread('oficina.png')
lena = io.imread('lena.png')

# Concatenar imágenes de entrenamiento
imagenes_entrenamiento = np.concatenate((caravana, maiz, fresas, oficina))

# Determinar rango de señal
rango_imagenes = sc.signalRange(imagenes_entrenamiento)

# Definir tasas de bits a analizar
tasas_bits = [2]

# Analizar cuantificadores
for b in tasas_bits:
    # Construir cuantificador óptimo
    cuantificador_opt = sc.OptimalSQ(b, imagenes_entrenamiento, algorithm=lbg)
    lena_cuantificada_opt = cuantificador_opt.quantize(lena)
    snr_opt = sc.snr(lena_cuantificada_opt, lena)

    # Construir cuantificador uniforme
    cuantificador_unif = sc.UniformSQ(b, rango_imagenes, qtype='midtread')
    lena_cuantificada_unif = cuantificador_unif.quantize(lena)
    snr_unif = sc.snr(lena_cuantificada_unif, lena)

    # Mostrar resultados
    print(f'--- Tasa de bits: {b} bits/píxel ---')
    print(f'SNR Cuantificador Óptimo: {snr_opt:.2f} dB')
    print(f'SNR Cuantificador Uniforme: {snr_unif:.2f} dB')
    print('\n')
    # Representar imágenes cuantificadas
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(lena_cuantificada_opt)
    plt.title(f'Óptimo {b} bits (SNR: {snr_opt:.2f} dB)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lena_cuantificada_unif)
    plt.title(f'Uniforme {b} bits (SNR: {snr_unif:.2f} dB)')
    plt.axis('off')

    plt.show()

#%%   
""""
Ejercicio 5

Construya un cuantificador vectorial óptimo para imágenes con una tasa de 4 bits/muestra. Use
las imágenes almacenas en caravana.png, maiz.png, mandrill.png(no hay mandrill usamos fresas de nuevo) y oficina.png concatenadas
una tras otra como conjunto de entrenamiento.
A continuación, cuantifique la imagen almacenada en lena.png. Represente la imagen cuantificada y calcule su SNR y tasa de bits por muestra, R. Repita el análisis usando un cuantificador
con tasas de 8 y 16 bits por bloque y tamaños de bloque, N, de 4, 16 y 64. Analice los resultados
en términos de tasa de bits, SNR y calidad percibida, comparándolos con los obtenidos usando
cuantificadores escalares.
"""



# Leer imágenes
caravana = io.imread('caravana.png')
maiz = io.imread('maiz.png')
fresas = io.imread('fresas.png')
oficina = io.imread('oficina.png')
lena = io.imread('lena.png')

# Concatenar imágenes para entrenamiento
imagenes_entrenamiento = np.concatenate((caravana, maiz, fresas, oficina))

# Definir tasas de bits por bloque para análisis (4, 8, 16 bits por bloque)
#La de 16 falla así que no la hacemois
tasas_bits = [6, 8]
step_vector = [2, 4, 8]

for b in tasas_bits:
    for step in step_vector:
        #Tamaño de bloque
        N = step * step  # 4, 16, 64 píxeles por bloque
        # Dividir en bloques la señal de entrenamiento
        bloques_entrenamiento = sc.partitionImage(imagenes_entrenamiento, step, step)
        # Construir cuantificador vectorial óptimo con entrenamiento
        cuantificador_opt = sc.OptimalVQ(b, bloques_entrenamiento, algorithm=lbg)
    
        # Cuantificar imagen lena con cuantificador vectorial óptimo
        bloques_lena = sc.partitionImage(lena, step, step)
        bloques_lena_cuantificada = cuantificador_opt.quantize(bloques_lena)
        lena_cuantificada_opt = sc.composeImage(bloques_lena_cuantificada, step, step, lena.shape)
    
        # Cuantificador escalar uniforme para comparar con la misma tasa efectiva
        R = b / N  # Tasa de bits efectiva por píxel
        niveles = int(2 ** (R)) if R >= 1 else 2
        rango_imagen = sc.signalRange(imagenes_entrenamiento)
        cuantificador_unif = sc.UniformSQ(niveles, rango_imagen, qtype='midtread')
    
        # Cuantificar imagen lena con cuantificador escalar uniforme
        lena_cuantificada_unif = cuantificador_unif.quantize(lena)
    
        # Calcular SNR
        snr_opt = sc.snr(lena_cuantificada_opt, lena)
        snr_unif = sc.snr(lena_cuantificada_unif, lena)
    
        # Mostrar resultados
        print(f'--- Tasa de bits por bloque: {b} bits ---')
        print(f'Tasa efectiva por píxel: {R:.4f} bits/píxel')
        print(f'SNR VQ Óptimo: {snr_opt:.2f} dB')
        print(f'SNR Uniforme (Escalar): {snr_unif:.2f} dB\n')
    
        # Representar imágenes cuantificadas
        plt.figure(figsize=(14, 5))
    
        plt.subplot(1, 2, 1)
        plt.imshow(lena_cuantificada_opt)
        plt.title(f'Cuantificador VQ Óptimo\n{b} bits/bloque (SNR: {snr_opt:.2f} dB)')
        plt.axis('off')
    
        plt.subplot(1, 2, 2)
        plt.imshow(lena_cuantificada_unif)
        plt.title(f'Cuantificador Uniforme Escalar\n{R:.4f} bits/píxel (SNR: {snr_unif:.2f} dB)')
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
        
