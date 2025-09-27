import numpy as np
import control as ct
import matplotlib.pyplot as plt
import sympy as sp
import json

#Matrices del sistema linealizado
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

A_num = np.array(A.subs(subs_dict).evalf(), dtype=float) #Sustituir y evaluar, convierte la matriz de sympy a un array de numpy para cálculos 
B_num = np.array(B.subs(subs_dict).evalf(), dtype=float)
C_num = np.array(C.evalf(), dtype=float)
D_num = np.array(D.evalf(), dtype=float)


#Crear sistema y pasarlo a función de transferencia
sys_ss = ct.ss(A_num, B_num, C_num, D_num) #Espacio de estados (representa el sistema lineal), describe la dinámica del sistema usando variables de estado
sys_tf = ct.ss2tf(sys_ss) #Convertir a función de transferencia


#Rango de ganancias para el root locus
kvec = np.linspace(0, 5000, 500) # Variación de ganancias k de 0 a 5000, espacio de 10 puntos
rlist, klist = ct.root_locus(sys_tf, gains=kvec, plot=False) 
#rlist = posiciones de polos en lazo cerrado para cada valor de k, klist = valores de k usados


#Graficar root locus completo
plt.figure(figsize=(8,6))
for i in range(rlist.shape[1]): #Cada columna de rlist corresponde a un polo
    plt.plot(np.real(rlist[:,i]), np.imag(rlist[:,i]), 'b-', linewidth=1) #Separamos la parte real y la imaginaria, ya que el formato de rlist es: parte_real + j*parte_imaginaria

#Polos de la planta (K=0)
poles_open = ct.poles(sys_tf) #Polos en lazo abierto
plt.plot(np.real(poles_open), np.imag(poles_open), 'ro', markersize=8, label="Polos K=0")


#Marcar polos en K específicos
K_especificos = [1, 10, 100, 1000] #Lista de valores de K para referencia
for K in K_especificos:
    sys_cl = ct.feedback(K*sys_tf, 1) #Sistema en lazo cerrado con ganancia K
    polos = ct.poles(sys_cl) #Polos en lazo cerrado para este K
    plt.plot(np.real(polos), np.imag(polos), 'x', markersize=10, label=f"K={K}")
    
    #Añadir etiquetas con el valor de K junto a las X
    for p in polos:
        plt.text(np.real(p)+0.0005, np.imag(p), f"K={K}", fontsize=8, color="black")


#Eje imaginario (frontera de estabilidad)
plt.axvline(0, color='k', linestyle='--', linewidth=0.8)

plt.title("Root Locus con posiciones específicas de K")
plt.xlabel("Parte real (σ)")
plt.ylabel("Parte imaginaria (jω)")
plt.grid(True)
plt.legend()
plt.show()



#Criterio de Routh-Hurwitz para sistema de segundo orden
denominador = sys_tf.den[0][0]  #Extrae los coeficientes del denominador s² + a₁s + a₀
a1, a0 = denominador[1], denominador[2]
print("\n2. CRITERIO DE ROUTH-HURWITZ:")
print(f"   Ecuación característica: s² + {a1:.6f}s + {a0:.6f} = 0")
print(f"   a1 = {a1:.6f} > 0? {'Sí' if a1 > 0 else 'No'}")
print(f"   a0 = {a0:.6f} > 0? {'Sí' if a0 > 0 else 'No'}")
if a1 > 0 and a0 > 0:
    print("   → SISTEMA ESTABLE (criterio de Routh-Hurwitz satisfecho)")
else:
    print("   → SISTEMA INESTABLE (criterio de Routh-Hurwitz no satisfecho)")

#Márgenes de estabilidad
print("\n3. MÁRGENES DE ESTABILIDAD:")
margen_ganancia, margen_fase, freq_gan, freq_fase = ct.margin(sys_tf) #Obtiene los margenes de ganancia y fase
print(f"   Margen de Ganancia: {margen_ganancia:.2f} dB")
print(f"   Margen de Fase: {margen_fase:.2f} grados")
print(f"   Frecuencia de cruce de ganancia: {freq_gan:.4f} rad/s")
print(f"   Frecuencia de cruce de fase: {freq_fase:.4f} rad/s")
  


plt.figure(figsize=(15, 10))

#Respuesta temporal

#Respuesta al escalón en lazo cerrado (muestra cómo responde el sistema cuando la entrada cambia instantáneamente de 0 a 1)

K = 80
#Sistema en lazo cerrado con ganancia K
sys_cl = ct.feedback(K*sys_tf, 1) #Multiplicamos la función de transferencia por K y cerramos el lazo con una retroalimentación unitaria (G, H), G = Planta, h = 1 (retroalimentación unitaria)

plt.figure(figsize=(8,6))
t_esc, y_esc = ct.step_response(sys_cl, T=np.linspace(0, 1000, 1000)) #0 a 1000 segundos, paso de 1 segundo
#t_esc = tiempos, y_esc = respuesta al escalón
plt.plot(t_esc, y_esc, 'b-', linewidth=2, label=f"K={K}")
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (Temperatura normalizada)')
plt.title(f'Respuesta al Escalón con K={K}')
plt.grid(True)
plt.legend()
plt.show()


#Respuesta al impulso
plt.subplot(2, 2, 2)
t_imp, y_imp = ct.impulse_response(sys_tf, T=np.linspace(0, 1000, 1000)) #Tiempo de 0 a 1000 segundos
plt.plot(t_imp, y_imp, 'r-', linewidth=2)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Respuesta al Impulso del Sistema Linealizado')
plt.grid(True)


#Diagrama de Bode 
#Definir rango de frecuencias
omega = np.logspace(-4, 1, 400)  # de 1e-4 a 10 rad/s

#Calcular bode
mag, phase, omega = ct.bode(sys_tf, omega, plot=False)

plt.figure(figsize=(8,6))

#Magnitud
plt.subplot(2,1,1)
plt.semilogx(omega, 20*np.log10(mag), 'b-', linewidth=2)
plt.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)  # línea en 0 dB
plt.ylabel('Magnitud [dB]')
plt.title('Diagrama de Bode - Magnitud')
plt.grid(True, which='both')
plt.ylim([-40, 20])  # opcional, para que el 0 quede más centrado

#Fase
plt.subplot(2,1,2)
plt.semilogx(omega, phase*180/np.pi, 'r-', linewidth=2)
plt.axhline(-180, color='k', linestyle='--', linewidth=0.8, alpha=0.7)  # línea en -180°
plt.xlabel('Frecuencia [rad/s]')
plt.ylabel('Fase [°]')
plt.title('Diagrama de Bode - Fase')
plt.grid(True, which='both')

plt.tight_layout()
plt.show()

# Diagrama de Nyquist 
plt.subplot(2, 2, 4) 
ct.nyquist_plot(sys_tf, color='b', linewidth=2) 
plt.title('Diagrama de Nyquist') 
plt.grid(True) 
# Marcar punto crítico (-1, 0j) en Nyquist 
plt.plot(-1, 0, 'ro', markersize=8, label='Punto crítico (-1, 0j)') 
plt.legend() 
plt.tight_layout() 
plt.show()
