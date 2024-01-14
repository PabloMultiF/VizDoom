import matplotlib.pyplot as plt
import torch

# La técnica de Generalized Advantage Estimation (GAE) considera no solo las recompensas directas obtenidas por el agente, 
# sino también los valores de los estados y una generalización sobre varios pasos en el tiempo para proporcionar 
# estimaciones más estables y equilibradas de las ventajas de las acciones, incluso en entornos con recompensas 
# inciertas o ruidosas.

def compute_gae(rewards, values, dones, gamma, lam=0.95):

    advantage = []
    gae = torch.tensor(0.0) # Inicialización de la ventaja generalizada acumulada

    for t in reversed(range(len(rewards)-1)):
            
        if dones[t]:
            gae = torch.tensor(0.0)
        
        delta_t = rewards[t] + (gamma * values[t+1] * int(not dones[t+1])) - values[t] # Diferencia temporal de ventajas
        gae = delta_t + gamma * lam * gae # Actualización de la ventaja generalizada acumulada

        advantage.append(gae)
            
    advantage.reverse()

    return advantage

def draw_plot(history_score, color = "red"):
    plt.title("Historial de Puntajes")
    plt.xlabel("Ciclos")
    plt.ylabel("Puntaje")
    plt.plot(history_score, color)
    plt.show()
