# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:40:28 2025

@author: daniel
"""

import scipy.io.wavfile as wf
import numpy as np

# Obtener ahora la estima **sin sesgo** de la autocorrelación usando el comando `correlate` (MatLab: `xcorr`) de numpy. Comparar gráficamente esta estima con la obtenida mediante el estimador sesgado de los apartados anteriores.

# Cargar la señal de audio
fs, x = wf.read("handel.wav")
x = x - np.mean(x)  # Eliminar el nivel

# Estimar la autocorrelación
R = np.correlate(x, x, mode="full") / len(x)
n = np.arange(-len(x) + 1, len(x))
