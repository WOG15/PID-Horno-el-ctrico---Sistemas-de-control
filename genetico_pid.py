import numpy as np
import json

with open("datos.json","r") as f:
    datos = json.load(f)

C1 = datos["C1"]
C2 = datos["C2"]
R12 = datos["R12"]
eta = datos["eta"]
Ta = datos["Ta"]
sigma = datos["sigma"]
eps = datos["eps"]
A1 = datos["A1"]
A2 = datos["A2"]
h0 = datos["h0"]
h1 = datos["h1"]
alpha = datos["alpha"]
T2aster = datos["T2aster"]  # referencia

def modelo_no_lineal(T1, T2, u_value):
    T1K = T1 + 273.15
    T2K = T2 + 273.15
    TaK = Ta + 273.15

    Qrad1 = eps * sigma * A1 * (T1K**4 - TaK**4)
    Qrad2 = eps * sigma * A2 * (T2K**4 - TaK**4)

    # convección no lineal
    h_val = h0 + h1 * abs(T2 - Ta)**alpha
    Qconv2 = h_val * A2 * (T2 - Ta)

    dT1 = (-(T1 - T2)/R12 + eta * u_value - Qrad1) / C1
    dT2 = ((T1 - T2)/R12 - Qconv2 - Qrad2) / C2

    return dT1, dT2

# Función de evaluación (fitness)
def evaluar_pid(params, tiempo_simulacion=1000, dt=0.1):
    """
    Evalúa qué tan bueno es un conjunto de parámetros PID (menor error)
    """
    Kp, Ki, Kd = params
    
    # Resetear simulación
    T1, T2 = Ta, Ta
    integral, prev_error = 0, 0
    error_history = []
    
    # Simular
    for t in np.arange(0, tiempo_simulacion, dt):
        error = T2aster - T2
        integral += error * dt
        derivative = (error - prev_error) / dt
        
        u = Kp * error + Ki * integral + Kd * derivative
        u = max(0, min(u, 2000))  # Saturación
        
        # Integrar modelo
        dT1, dT2 = modelo_no_lineal(T1, T2, u)
        T1 += dT1 * dt
        T2 += dT2 * dt
        
        error_history.append(abs(error))
        prev_error = error
    
    # Calcular métricas de desempeño
    iae = np.trapz(np.abs(error_history), dx=dt)  #IAE (Integral of Absolute Error) ∫∣e(t)∣dt mide el error acumulado en el tiempo.
    itae = np.trapz(np.arange(len(error_history)) * np.abs(error_history), dx=dt)  #ITAE (Integral of Time-weighted Absolute Error) = ∫t∣e(t)∣dt, penaliza más los errores que ocurren tarde (penaliza una estabilización tardía)
    #Trapz integración numérica
    return iae + 0.1 * itae  #Función de costo combinada = IAE + 0.1 * ITAE = fitness (el valor menor de la suma es mejor)

# Algoritmo Genético
class AlgoritmoGenetico:
    def __init__(self, poblacion_size=50, generations=100):
        self.poblacion_size = poblacion_size
        self.generations = generations
        
        # Límites de búsqueda para Kp, Ki, Kd
        self.bounds = [(1, 500), (0.001, 2.0), (10, 1000)]
    
    def inicializar_poblacion(self): #Para cada par (low, high) en los limites, genera un vector con self.poblacion_size números aleatorios uniformes (mismo chance) entre esos valores
        return np.array([np.random.uniform(low, high, self.poblacion_size) 
                        for (low, high) in self.bounds]).T #Transponer para tener individuos como filas en vez de filas 1.kp, 2.ki, 3.kd
    
    def seleccion(self, poblacion, scores):
        #Selección por torneo
        selected = []
        for _ in range(self.poblacion_size): #Repetimos hasta llenar la nueva población (una iteración por individuo)
            idx = np.random.randint(0, len(poblacion), 3) #Seleccionamos 3 individuos al azar (indices de la población)
            best_idx = idx[np.argmin(scores[idx])] #Selecciona el mejor de los 3 (menor score)
            selected.append(poblacion[best_idx])
        return np.array(selected)
    
    def cruza(self, padre1, padre2):
        # Cruza uniforme
        mask = np.random.random(3) < 0.5 #convierte kp, ki, kd en 3 numeros aleatorios entre 0 y 1, si <0.5  es True
        hijo = padre1.copy()
        hijo[mask] = padre2[mask] #Si mask = True, toma el valor del padre2, si false, del padre1
        return hijo
    
    def mutacion(self, individuo, mutation_rate=0.1):
        #Mutación gaussiana
        for i in range(3): #Recorre cada gen del individuo (kp, ki,kKd)
            if np.random.random() < mutation_rate: #10% probabilidad de mutción, si el numero random < mutation_rate, muta
                low, high = self.bounds[i] #Rango
                individuo[i] += np.random.normal(0, (high-low)/10) #i=0 kp, i=1 ki, i=2 kd, se le suma un valor pequeño random normal con media 0 y desviación estándar (high-low)/10
                individuo[i] = np.clip(individuo[i], low, high) #Evita que el nuevo valor se salga de los límites definidos
        return individuo
    
    def ejecutar(self):
        poblacion = self.inicializar_poblacion()
        mejor_global = None
        mejor_score = float('inf')
        
        for gen in range(self.generations):
            # Evaluar
            scores = np.array([evaluar_pid(ind) for ind in poblacion])
            
            # Actualizar mejor global
            best_idx = np.argmin(scores)
            if scores[best_idx] < mejor_score:
                mejor_score = scores[best_idx]
                mejor_global = poblacion[best_idx].copy()
            
            # Evolución
            seleccionados = self.seleccion(poblacion, scores)
            nueva_poblacion = []
            
            for i in range(0, self.poblacion_size, 2):
                padre1, padre2 = seleccionados[i], seleccionados[i+1]
                hijo1 = self.mutacion(self.cruza(padre1, padre2))
                hijo2 = self.mutacion(self.cruza(padre2, padre1))
                nueva_poblacion.extend([hijo1, hijo2])
            
            poblacion = np.array(nueva_poblacion)
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Mejor score = {mejor_score:.4f}, Parámetros = {mejor_global}")
        
        return mejor_global, mejor_score

# 3. Ejecutar optimización
if __name__ == "__main__":
    ag = AlgoritmoGenetico(poblacion_size=30, generations=50)
    mejor_params, mejor_score = ag.ejecutar()
    
    print(f"\nMEJORES PARÁMETROS ENCONTRADOS:")
    print(f"Kp = {mejor_params[0]:.3f}")
    print(f"Ki = {mejor_params[1]:.4f}") 
    print(f"Kd = {mejor_params[2]:.3f}")
    print(f"Función de costo = {mejor_score:.4f}")