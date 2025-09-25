import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import json
with open("datos.json", "r") as f:
    params = json.load(f)

#Parámetros

C1 = params["C1"]   # J/K (capacidad térmica elemento)
C2 = params["C2"]     # J/K (capacidad térmica aire interior)
R12 = params["R12"]      # K/W (resistencia térmica entre T1 y T2)
eta = params["eta"]      # eficiencia de la resistencia
Ta = params["Ta"]      # °C temperatura ambiente
pwr = params["pwr"]   # potencia aplicada en W

#Radiación
sigma = params["sigma"]  # Stefan-Boltzmann
eps = params["eps"]              # emisividad
A1 = params["A1"]               # m² área radiación masa 1
A2 = params["A2"]                # m² área radiación masa 2

#Convección (no lineal)
h0 = params["h0"]     # coeficiente base [W/m²K]
h1 = params["h1"]   # coeficiente adicional
alpha = params["alpha"]  # exponente (lineal en deltaT)


#Modelo
def deriv(t, x):
    T1, T2 = x

    #Pasar a Kelvin para radiación
    T1K = T1 + 273.15
    T2K = T2 + 273.15
    TaK = Ta + 273.15

    #Pérdidas por radiación
    Qrad1 = eps * sigma * A1 * (T1K**4 - TaK**4)
    Qrad2 = eps * sigma * A2 * (T2K**4 - TaK**4)

    #Convección (no lineal)
    h = h0 + h1 * abs(T2 - Ta)**alpha
    Qconv2 = h * A2 * (T2 - Ta)

    #Ecuaciones diferenciales
    dT1 = (-(T1 - T2)/R12 + eta * pwr - Qrad1) / C1
    dT2 = ((T1 - T2)/R12 - Qconv2 - Qrad2) / C2

    return [dT1, dT2] #derivadas (cambio en el tiempo)


#Simulación
x0 = [Ta, Ta]  # condiciones iniciales
t_final = 3600 # 1 hora
t_eval = np.linspace(0, t_final, 601) #3600/601 = 6s de paso

sol = solve_ivp(deriv, [0, t_final], x0, t_eval=t_eval) #método de integración numérica para resolver en el tiempo el sistema no lineal (Rounge-Kutta)

T1, T2 = sol.y
t = sol.t

#Valores finales
print("Temperatura final T1 (elemento):", round(T1[-1], 2), "°C")
print("Temperatura final T2 (interior):", round(T2[-1], 2), "°C")

#Grafica
plt.figure(figsize=(8,4))
plt.plot(t, T1, label="T1 (elemento)")
plt.plot(t, T2, label="T2 (interior)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Temperatura (°C)")
plt.title(f"Modelo NO lineal con radiación + convección\nP = {pwr} W")
plt.legend()
plt.grid(True)

#Guardar la figura
plt.savefig("Sim. modelo no lineal.png", dpi=300, bbox_inches='tight')    
plt.show()


#Punto de equilibrio 

def equilibrio(x0):
    T1, T2 = x0

    #Pasar a Kelvin para radiación
    T1K = T1 + 273.15
    T2K = T2 + 273.15
    TaK = Ta + 273.15

    #Pérdidas por radiación
    Qrad1 = eps * sigma * A1 * (T1K**4 - TaK**4)
    Qrad2 = eps * sigma * A2 * (T2K**4 - TaK**4)

    #Convección (no lineal)
    h = h0 + h1 * abs(T2 - Ta)**alpha
    Qconv2 = h * A2 * (T2 - Ta)

    #Ecuaciones diferenciales
    f1 = -(T1 - T2)/R12 + eta * pwr - Qrad1 #flujo de calor, in - out = 0 
    f2 = (T1 - T2)/R12 - Qconv2 - Qrad2

    return [f1, f2]

sol, info, ier, msg = fsolve(equilibrio, x0, full_output=True) #encuentra la raíz (punto de equilibrio)
T1aster, T2aster = sol

print("Convergencia:", "Si" if ier==1 else "No")
print("Mensaje:", msg)
print(f"u* = {pwr:.1f} W")
print(f"T1* = {T1aster:.2f} °C, T2* = {T2aster:.2f} °C")

params["T1aster"] = T1aster
params["T2aster"] = T2aster 
params["uaster"] = pwr

with open("datos.json", "w") as f:
    json.dump(params, f, indent=4)