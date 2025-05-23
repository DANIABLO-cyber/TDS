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
figsize = (12, 6)
plt.figure()
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
figsize = (12, 6)

plt.figure()
figsize = (12, 6)
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
figsize = (12, 6)

plt.figure()
plt.plot(p_out_15, gain_15, marker="o", label="15 mW")
plt.plot(p_out_35, gain_35, marker="s", label="35 mW")
plt.plot(p_out_50, gain_50, marker="x", label="50 mW")
plt.plot(p_out_70, gain_70, marker="^", label="70 mW")
plt.title("Ganancia vs Potencia de salida")
plt.xlabel("Potencia de salida (dBm)")
plt.ylabel("Ganancia (dB)")
plt.legend()
plt.grid(True)

# 4) Curva teórica (Ecuación 4)
figsize = (12, 6)

diff = np.linspace(-45, 15, 200)
theoretical_gain_eq4 = 0.99932 * diff
plt.figure()
plt.plot(diff, theoretical_gain_eq4)
plt.title("Curva teórica (Ecuación 4)")
plt.xlabel("P_out - P_in (dB)")
plt.ylabel("Ganancia (dB)")
plt.grid(True)

# 5) Ganancia pequeña señal vs potencia de bombeo (Ecuación 3)

pump = np.array([15, 35, 50, 70])  # mW
g_ss = np.array([gain_15[0], gain_35[0], gain_50[0], gain_70[0]])  # a P_in = -40 dBm
gamma_l = g_ss / 4.34  # γ0 * l estimado
pumping_range = np.linspace(10, 80, 100)
gain_theoretical_eq3 = 4.34 * np.interp(pumping_range, pump, gamma_l)


plt.figure()
figsize = (12, 6)
plt.plot(pumping_range, gain_theoretical_eq3)
plt.scatter(pump, g_ss, marker="o", label="Puntos medidos")
plt.title("Ganancia pequeña señal vs Potencia de bombeo (Ecuación 3)")
plt.xlabel("Potencia de bombeo (mW)")
plt.ylabel("Ganancia pequeña señal (dB)")
plt.legend()
plt.grid(True)

# 6) Potencia de saturación vs potencia de bombeo (Figura 8)
p_sat_3dB = np.array([-2, 4.7, 6, 7.5])
p_sat_10dB = np.array([2, 6.5, 9, 10.5])

figsize = (12, 6)

plt.figure()
plt.plot(pump, p_sat_3dB, marker="o", label="-3 dB")
plt.plot(pump, p_sat_10dB, marker="s", label="-10 dB")
plt.title("Potencia de saturación vs Potencia de bombeo")
plt.xlabel("Potencia de bombeo (mW)")
plt.ylabel("P_sat (dBm)")
plt.legend()
plt.grid(True)

plt.show()
