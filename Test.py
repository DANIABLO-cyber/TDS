# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:40:28 2025

@author: daniel
"""

import scipy.io.wavfile as wf
import numpy as np


def x(n):
    if n == 0:
        return 1
    elif n > 0:
        return 1.9 * (0.9 ** (n - 1))
    else:
        return 0


# Calcular r_x(1) = sum_{n=1}^{\infty} x(n) * x(n - 1)
# Se trunca la suma cuando los términos sean suficientemente pequeños
rx1 = 0
tolerance = 1e-10
n = 1

while True:
    term = x(n) * x(n - 1)
    if abs(term) < tolerance:
        break
    rx1 += term
    n += 1

print(f"r_x(1) ≈ {rx1}")
