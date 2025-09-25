import json
import sympy as sp
import numpy as np

import control as ct
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

with open("datos.json","r") as f:
    datos = json.load(f)

C1 = datos["C1"]
C2 = datos["C2"]
R12 = datos["R12"]
eta = datos["eta"]
Ta = datos["Ta"]
u_aster = datos["uaster"]
sigma = datos["sigma"]
eps = datos["eps"]
A1 = datos["A1"]
A2 = datos["A2"]
h0 = datos["h0"]
h1 = datos["h1"]
alpha = datos["alpha"]
T1aster = datos["T1aster"]
T2aster = datos["T2aster"]

#Variables simbólicas
T1, T2, u = sp.symbols("T1 T2 u")

#Matrices 
A = sp.Matrix([ #matriz usando SymPy, que puede manejar símbolos y operaciones algebraicas
    [-(4*A1*(T1 + 273.15)**3*eps*sigma + 1/R12)/C1,  1/(C1*R12)],
    [ 1/(C2*R12), -(4*A2*(T2 + 273.15)**3*eps*sigma + 
                    A2*(T2 - Ta)**(alpha - 1)*(T2 - Ta)*alpha*h1 + 
                    ((T2 - Ta)**alpha*h1 + h0)*A2 + 1/R12)/C2]
])

B = sp.Matrix([
    [eta/C1],
    [0]
])

C = sp.Matrix([[0,1]])
D = sp.Matrix([[0]])

#Sustituir valores de equilibrio
subs_dict = {
    T1: T1aster,
    T2: T2aster,
    u: u_aster
}

A_num = np.array(A.subs(subs_dict).evalf(), dtype=float) #Sustituimos por las variables simbolicas y evaluamos las expresiones a númers, convierte la matriz de sympy a un array de numpy para cálculos 
B_num = np.array(B.subs(subs_dict).evalf(), dtype=float) 
C_num = np.array(C.evalf(), dtype=float)
D_num = np.array(D.evalf(), dtype=float)

#Resultados
print("Matriz A:")
print(A_num)
print("\nMatriz B:")
print(B_num)
print("\nMatriz C:")
print(C_num)
print("\nMatriz D:")
print(D_num)

print("\nModelo linealizado:")
print("x_dot = A * x_tilde + B * u_tilde")
print("y     = C * x_tilde + D * u_tilde")

#Crear sistema lineal con python-control
sys_lin = ct.ss(A_num, B_num, C_num, D_num) #Crea un objeto de espacio de estados, con las matrices evaluadas (representa el sistema lineal)

#Modelo no lineal
def deriv_no_lineal(t, x, u_value):
    T1, T2 = x
    
    #Pasar a Kelvin para radiación
    T1K = T1 + 273.15
    T2K = T2 + 273.15
    TaK = Ta + 273.15

    #Pérdidas por radiación
    Qrad1 = eps * sigma * A1 * (T1K**4 - TaK**4)
    Qrad2 = eps * sigma * A2 * (T2K**4 - TaK**4)

    #Convección no lineal
    h = h0 + h1 * abs(T2 - Ta)**alpha
    Qconv2 = h * A2 * (T2 - Ta)

    #Ecuaciones diferenciales
    dT1 = (-(T1 - T2)/R12 + eta * u_value - Qrad1) / C1
    dT2 = ((T1 - T2)/R12 - Qconv2 - Qrad2) / C2

    return [dT1, dT2]


#Simulación comparativa con perturbación

#Condiciones iniciales (punto de equilibrio + pequeña perturbación)
perturbacion = 5.0  #desviación del equilibrio
x0_nolineal = [T1aster + perturbacion, T2aster]  #perturbamos solo T1 (fuente de calor)
x0_lineal = [perturbacion, 0]  #Para el modelo lineal: [T1~, T2~] = [5, 0] (desviaciones)

#Tiempo de simulación
t_final = 300  #5 minutos 
t_eval = np.linspace(0, t_final, 1000) #paso de 0.3 s

#Entrada constante (mantenemos la potencia de equilibrio)
u_sim = u_aster

#Simulación del modelo NO LINEAL
sol_nolineal = solve_ivp(
    deriv_no_lineal, 
    [0, t_final], 
    x0_nolineal, 
    t_eval=t_eval, 
    args=(u_sim,)
)
T1_nolineal, T2_nolineal = sol_nolineal.y

#Simulación del modelo LINEAL
#Para el modelo lineal: ẋ = A*x + B*u
#Como estamos en desviaciones (x_tilde), la entrada u_tilde = 0 (pues u_sim = u_aster)
t_lineal, y_lineal = ct.initial_response(sys_lin, T=t_eval, X0=x0_lineal) #simula el comportamiento del sistema solo por la perturbación inicial

#Convertir salida del modelo lineal a temperatura absoluta
T2_lineal = y_lineal + T2aster  #quitarle la desviación, y lineal es la salida

#Grafica comparativa
plt.figure(figsize=(12, 8))

#Gráfica de T2 (temperatura interior - la salida y)
plt.subplot(2, 1, 1)
plt.plot(t_eval, T2_nolineal, 'b-', linewidth=2, label='Modelo no lineal (T2)')
plt.plot(t_lineal, T2_lineal, 'r--', linewidth=2, label='Modelo lineal (T2)')
plt.axhline(y=T2aster, color='k', linestyle=':', alpha=0.7, label='Equilibrio T2*')
plt.ylabel('Temperatura [°C]')
plt.title('Comparación: Respuesta a perturbación inicial\n(Modelo no lineal vs Linealizado)')
plt.legend()
plt.grid(True)

plt.savefig("Comparacion_modelo lineal vs no lineal.png", dpi=300, bbox_inches='tight')
plt.show()

with open("datos.json", "r") as f:
    params = json.load(f)

params["A_num"] = A_num.tolist()
params["B_num"] = B_num.tolist()
params["C_num"] = C_num.tolist()
params["D_num"] = D_num.tolist()

with open("datos.json", "w") as f:
    json.dump(params, f, indent=4)