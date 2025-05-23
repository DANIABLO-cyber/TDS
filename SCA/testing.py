import numpy as np
import matplotlib.pyplot as plt

# Datos experimentales aproximados a partir de las figuras
input_pwr = np.array([-5, -10, -15, -20, -25, -30, -35, -40])


# Datos del fotoreceptor para cada potencia de bombeo
p_out_15 = np.array([-10.6, -11.8, -13.8, -16.9, -21.3, -26.2, -31.1, -36.1])
p_out_35 = np.array([-5.4, -6.5, -8.7, -12.3, -16.8, -21.7, -26.7, -31.9])
p_out_50 = np.array([-3.6, -4.9, -7.2, -11.0, -15.5, -20.3, -25.5, -30.7])
p_out_70 = np.array([-2.2, -3.4, -5.9, -9.9, -14.4, -19.3, -24.3, -29.6])


# Cálculo de ganancia según ecuación (4): Gain(dB) = 0.99932 * (P_out - P_in)
gain_15 = p_out_15 - input_pwr
gain_35 = p_out_35 - input_pwr
gain_50 = p_out_50 - input_pwr
gain_70 = p_out_70 - input_pwr


# 1) Potencia de salida vs potencia de entrada (Figura 5)
plt.figure(figsize=(12, 6))
plt.plot(input_pwr, p_out_15, marker="o", label="15 mW")
plt.plot(input_pwr, p_out_35, marker="s", label="35 mW")
plt.plot(input_pwr, p_out_50, marker="x", label="50 mW")
plt.plot(input_pwr, p_out_70, marker="^", label="70 mW")
plt.title("Potencia de salida vs Potencia de entrada")
plt.xlabel("Potencia de entrada (dBm)")
plt.ylabel("Potencia de salida (dBm)")
plt.legend()
plt.grid(True)

# 2) Ganancia vs potencia de entrada (Figura 6)
plt.figure(figsize=(12, 6))
plt.plot(input_pwr, gain_15, marker="o", label="15 mW")
plt.plot(input_pwr, gain_35, marker="s", label="35 mW")
plt.plot(input_pwr, gain_50, marker="x", label="50 mW")
plt.plot(input_pwr, gain_70, marker="^", label="70 mW")
plt.title("Ganancia vs Potencia de entrada")
plt.xlabel("Potencia de entrada (dBm)")
plt.ylabel("Ganancia (dB)")
plt.legend()
plt.grid(True)

# 3) Ganancia vs potencia de salida (Figura 7)
plt.figure(figsize=(12, 6))
plt.plot(p_out_15, gain_15, marker="o", label="15 mW")
plt.plot(p_out_35, gain_35, marker="s", label="35 mW")
plt.plot(p_out_50, gain_50, marker="x", label="50 mW")
plt.plot(p_out_70, gain_70, marker="^", label="70 mW")
plt.title("Ganancia vs Potencia de salida")
plt.xlabel("Potencia de salida (dBm)")
plt.ylabel("Ganancia (dB)")
plt.legend()
plt.grid(True)
