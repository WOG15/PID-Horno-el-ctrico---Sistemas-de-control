import pygame
import numpy as np
import json

# Cargar datos desde JSON
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

# Modelo no lineal (dinámica completa)
def modelo_no_lineal(T1, T2, u_value):
    T1K = T1 + 273.15
    T2K = T2 + 273.15
    TaK = Ta + 273.15

    Qrad1 = eps * sigma * A1 * (T1K**4 - TaK**4)
    Qrad2 = eps * sigma * A2 * (T2K**4 - TaK**4)
    h_val = (h0 + h1 * abs(T2 - Ta)**alpha) * conv_factor  #factor de convección ajustable
    Qconv2 = h_val * A2 * (T2 - Ta)

    dT1 = (-(T1 - T2)/R12 + eta * u_value - Qrad1) / C1
    dT2 = ((T1 - T2)/R12 - Qconv2 - Qrad2) / C2

    return dT1, dT2

# Configuración PID inicial

# Configuración PID inicial
Kp_manual, Ki_manual, Kd_manual = 80.0, 0.1, 500.0  # Valores manuales
Kp_genetic = datos["Kpg"]  # Valores del algoritmo genético
Ki_genetic = datos["Kig"]
Kd_genetic = datos["Kdg"]

# Inicializar con valores manuales
Kp, Ki, Kd = Kp_manual, Ki_manual, Kd_manual
using_genetic = False

integral_error, prev_error = 0.0, 0.0
conv_factor = 1.0  # Factor multiplicador para convección

# Inicializar pygame
pygame.init()
WIDTH, HEIGHT = 1200, 800  # Ventana 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación PID - Tres Gráficas Separadas")

font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 20, bold=True)

# Función para dibujar ejes y etiquetas
def draw_axes(screen, rect, x_label, y_label, min_val, max_val, n_divisions=5):
    # Eje Y
    pygame.draw.line(screen, (0,0,0), (rect[0]+40, rect[1]+10), (rect[0]+40, rect[1]+rect[3]-20), 1)
    # Eje X
    pygame.draw.line(screen, (0,0,0), (rect[0]+40, rect[1]+rect[3]-20), (rect[0]+rect[2]-10, rect[1]+rect[3]-20), 1)
    # Etiquetas de ejes
    label = font.render(x_label, True, (0,0,0))
    screen.blit(label, (rect[0]+rect[2]-60, rect[1]+rect[3]-15))
    label = font.render(y_label, True, (0,0,0))
    screen.blit(label, (rect[0]+10, rect[1]+10))
    # Marcas y valores en eje Y
    for i in range(n_divisions+1):
        
        # Calcula la posición vertical de cada marca
        y = rect[1] + rect[3] -20 - (i * (rect[3]-30) / n_divisions)
        # Dibuja la línea de marca 
        pygame.draw.line(screen, (0,0,0), (rect[0]+37, y), (rect[0]+43, y), 1)
        # Calcula el valor numérico para esta marca
        val = min_val + (i * (max_val - min_val) / n_divisions)
         # Renderiza y coloca el texto del valor
        label = font.render(f"{val:.0f}", True, (0,0,0))
        screen.blit(label, (rect[0]+10, y-8))

# Variables de simulación 
dt = 0.3
T1_actual, T2_actual = Ta, Ta
r = T2aster  # Referencia = temperatura de equilibrio
time = 0

# Historial para las tres gráficas
t_hist, y_hist, T1_hist, u_hist, error_hist = [], [], [], [], []

# Variables para cajas de entrada
input_boxes = {
    'Kp': {'rect': pygame.Rect(50, 650, 100, 30), 'text': str(Kp), 'active': False},
    'Ki': {'rect': pygame.Rect(200, 650, 100, 30), 'text': str(Ki), 'active': False},
    'Kd': {'rect': pygame.Rect(350, 650, 100, 30), 'text': str(Kd), 'active': False},
    'Ref': {'rect': pygame.Rect(500, 650, 100, 30), 'text': str(T2aster), 'active': False},
    'Conv': {'rect': pygame.Rect(650, 650, 100, 30), 'text': str(conv_factor), 'active': False},
    'C1': {'rect': pygame.Rect(50, 700, 100, 30), 'text': str(C1), 'active': False},
    'C2': {'rect': pygame.Rect(200, 700, 100, 30), 'text': str(C2), 'active': False},
    'A1': {'rect': pygame.Rect(350, 700, 100, 30), 'text': str(A1), 'active': False},
    'A2': {'rect': pygame.Rect(500, 700, 100, 30), 'text': str(A2), 'active': False}
}

#Intercambio valores manual/genético
input_boxes['Toggle'] = {
    'rect': pygame.Rect(800, 650, 150, 30),
    'text': "Cambiar a PID Genético" if not using_genetic else "Usar PID Manual",
    'active': False
}

# Bucle principal 
running = True
while running:
    clock = pygame.time.Clock()
    clock.tick(120)

    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Manejo de cajas de entrada
        if event.type == pygame.MOUSEBUTTONDOWN:
            for key, box in input_boxes.items():
                if box['rect'].collidepoint(event.pos):
                    if key == 'Toggle':
                        using_genetic = not using_genetic
                        if using_genetic:
                            Kp, Ki, Kd = Kp_genetic, Ki_genetic, Kd_genetic
                            box['text'] = "Usar PID Manual"
                        else:
                            Kp, Ki, Kd = Kp_manual, Ki_manual, Kd_manual
                            box['text'] = "Usar PID Genético"
                        # Actualizar los textos de las cajas
                        input_boxes['Kp']['text'] = f"{Kp}"
                        input_boxes['Ki']['text'] = f"{Ki}"
                        input_boxes['Kd']['text'] = f"{Kd}"
                        integral_error = 0.0  # Resetear integral
                    else:
                        box['active'] = True
                else:
                    box['active'] = False
        
        # Entrada de texto de las cajas
        if event.type == pygame.KEYDOWN:
            for key, box in input_boxes.items():
                if box['active']:
                    if event.key == pygame.K_RETURN:
                        # Aplicar nuevos valores
                        try:
                            if key == 'Kp': Kp = float(box['text'])
                            elif key == 'Ki': Ki = float(box['text'])
                            elif key == 'Kd': Kd = float(box['text'])
                            elif key == 'Ref': r = float(box['text'])
                            elif key == 'Conv': conv_factor = float(box['text'])
                            elif key == 'C1': C1 = float(box['text'])
                            elif key == 'C2': C2 = float(box['text'])
                            elif key == 'A1': A1 = float(box['text'])
                            elif key == 'A2': A2 = float(box['text'])
                            # Resetear integral al cambiar referencia para evitar el wind-up (si hay un cambio grande en la referencia, el error acumulado puede ser muy grande (integral) puede volverse muy grande)
                            integral_error = 0.0
                        except ValueError:
                            pass
                        box['active'] = False
                    elif event.key == pygame.K_BACKSPACE:
                        box['text'] = box['text'][:-1]
                    else:
                        box['text'] += event.unicode

    # Cálculo del PID
    error = r - T2_actual # Error actual
    integral_error += error * dt # Acumulación del error en el tiempo (forma númerica aproximada)
    derivative = (error - prev_error) / dt if dt > 0 else 0 #Tasa de cambio del error (derivada aproximada (diferencia finita entre el error actual y el anterior)
    u = Kp * error + Ki * integral_error + Kd * derivative
    prev_error = error

    # Aplicar límites físicos
    u = max(0, min(u, 2000))

    # Integrar modelo NO LINEAL
    dT1, dT2 = modelo_no_lineal(T1_actual, T2_actual, u)
    T1_actual += dT1 * dt #Temperatura del momento siguiente
    T2_actual += dT2 * dt
    T1_actual = max(Ta, T1_actual)
    T2_actual = max(Ta, T2_actual)

    # Guardar historial
    time += dt
    t_hist.append(time)
    y_hist.append(T2_actual)
    T1_hist.append(T1_actual)
    u_hist.append(u)
    error_hist.append(error)
    
    # Mantener ventana de tiempo
    if len(t_hist) > 400:
        t_hist.pop(0)
        y_hist.pop(0)
        T1_hist.pop(0)
        u_hist.pop(0)
        error_hist.pop(0)

    # Dibujar
    screen.fill((240, 240, 240))  # Fondo gris claro

    # Gráfica: Estados x(t) 
    graph1_rect = (50, 50, 350, 180)
    pygame.draw.rect(screen, (255, 255, 255), graph1_rect)
    title = title_font.render("Estados x(t): T1 y T2", True, (0, 0, 0))
    screen.blit(title, (60, 55))
    temp_min = Ta  # Mínimo fijo
    temp_max = max(T2aster * 1.5, 150)  # Máximo fijo
    draw_axes(screen, graph1_rect, "t[s]", "T[°C]", temp_min, temp_max)
    x_points = np.linspace(90, 340, len(t_hist))  # Comienza a la derecha del eje
    
    if len(t_hist) > 1:
        # T2 (temperatura controlada)
        y2_points = 210 - ((np.array(y_hist) - temp_min) / (temp_max - temp_min)) * 150
        pygame.draw.lines(screen, (0, 0, 255), False, list(zip(x_points, y2_points)), 2)
        
        # T1 (elemento calefactor)
        y1_points = 210 - ((np.array(T1_hist) - temp_min) / (temp_max - temp_min)) * 150
        pygame.draw.lines(screen, (255, 165, 0), False, list(zip(x_points, y1_points)), 2)
        
        # Referencia
        ref_y = 210 - ((r - temp_min) / (temp_max - temp_min)) * 150
        pygame.draw.line(screen, (255, 0, 0), (60, ref_y), (340, ref_y), 1)

    # Gráfica: Error e(t) 
    graph2_rect = (450, 50, 350, 180)
    pygame.draw.rect(screen, (255, 255, 255), graph2_rect)
    title = title_font.render("Error e(t) = r - T2", True, (0, 0, 0))
    screen.blit(title, (460, 55))
    error_max = 50  # Escala fija +-50°C
    draw_axes(screen, graph2_rect, "t[s]", "e[°C]", -error_max, error_max)
    x_points = np.linspace(490, 790, len(t_hist))
    
    if len(t_hist) > 1:
        error_points = 210 - (np.array(error_hist) / (2 * error_max) + 0.5) * 150
        pygame.draw.lines(screen, (255, 0, 0), False, list(zip(x_points, error_points)), 2)
        # Línea de error cero
        zero_y = 210 - (0.5) * 150
        pygame.draw.line(screen, (0, 0, 0), (460, zero_y), (790, zero_y), 1)

    # Gráfica: Entrada u(t) 
    graph3_rect = (850, 50, 300, 180)
    pygame.draw.rect(screen, (255, 255, 255), graph3_rect)
    title = title_font.render("Entrada u(t) - Potencia", True, (0, 0, 0))
    screen.blit(title, (860, 55))
    u_max = 2000  # Escala fija de potencia
    draw_axes(screen, graph3_rect, "t[s]", "u[W]", 0, u_max)
    x_points = np.linspace(890, 1130, len(t_hist))
    
    if len(t_hist) > 1:
        u_points = 210 - (np.array(u_hist) / u_max) * 150
        pygame.draw.lines(screen, (0, 150, 0), False, list(zip(x_points, u_points)), 2)

    #Cajas de entrada
    pygame.draw.rect(screen, (255, 255, 255), (50, 600, 750, 100))  
    title = title_font.render("Parámetros del PID y Factor de Convección", True, (0, 0, 0))
    screen.blit(title, (60, 610))
    
    labels = ['Kp:', 'Ki:', 'Kd:', 'Referencia:', 'Factor Conv:', 'C1:', 'C2:', 'A1:', 'A2:', "Toggle PID:"]
    for i, (key, box) in enumerate(input_boxes.items()):
        # Etiqueta
        label = font.render(labels[i], True, (0, 0, 0))
        screen.blit(label, (box['rect'].x - 40, box['rect'].y + 8))
        
        # Caja de entrada
        color = (100, 100, 255) if box['active'] else (200, 200, 200)
        pygame.draw.rect(screen, color, box['rect'])
        pygame.draw.rect(screen, (0, 0, 0), box['rect'], 2)
        
        # Texto
        text_surface = font.render(box['text'], True, (0, 0, 0))
        screen.blit(text_surface, (box['rect'].x + 5, box['rect'].y + 8))

    #Datos en tiempo real
    info_box = pygame.Rect(50, 400, 450, 100)
    pygame.draw.rect(screen, (255, 255, 255), info_box)
    pygame.draw.rect(screen, (0, 0, 0), info_box, 2)
    
    info_text = [
        f"Tiempo: {time:.1f} s",
        f"T2 actual: {T2_actual:.2f}°C, Referencia: {r:.2f}°C",
        f"Error actual: {error:.2f}°C, Potencia: {u:.1f}W",
        f"T1 (elemento): {T1_actual:.2f}°C"
    ]
    
    for i, text in enumerate(info_text):
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (info_box.x + 10, info_box.y + 10 + i*20))

    pygame.display.flip()

pygame.quit()