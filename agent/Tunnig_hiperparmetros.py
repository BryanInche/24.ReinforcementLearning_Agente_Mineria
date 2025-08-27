# sensitivity_analysis.py
import pandas as pd
from multiprocessing import Pool
from Tunnig_agent_train_base import entrenar_configuracion_analisisSensibilidad  # tu función ya existente

# === Definir hiperparámetros a probar ===
alphas = [0.1, 0.3, 0.5]
gammas = [0.8, 0.9, 0.99]
epsilon_decays = [0.99, 0.995, 0.999]

# Generamos todas las combinaciones (un factor a la vez)
# Nota: aquí estamos haciendo un análisis "uno a la vez":
#   - variamos alpha dejando gamma y decay fijos

# -------------------------------
# 3) CONSTRUIR LA LISTA DE EXPERIMENTOS
# -------------------------------
# En este punto, "params" es una lista de tuplas con 4 campos:
#  (nombre_del_parametro_que_varía, alpha, gamma, epsilon_decay)
"""
# params = [A, B, C, D, E, F, G, H, I] (9 configuraciones)
# Variamos ALPHA (gamma y decay fijos)
("alpha", 0.1, 0.9, 0.995)
("alpha", 0.3, 0.9, 0.995) 
("alpha", 0.5, 0.9, 0.995)

# Variamos GAMMA (alpha y decay fijos)
("gamma", 0.1, 0.8, 0.995)
("gamma", 0.1, 0.9, 0.995)
("gamma", 0.1, 0.99, 0.995)

# Variamos DECAY (alpha y gamma fijos)
("epsilon_decay", 0.1, 0.9, 0.99)
("epsilon_decay", 0.1, 0.9, 0.995) 
("epsilon_decay", 0.1, 0.9, 0.999)
"""
params = []
for a in alphas:
    params.append(("alpha", a, 0.9, 0.995))
for g in gammas:
    params.append(("gamma", 0.1, g, 0.995))
for e in epsilon_decays:
    params.append(("epsilon_decay", 0.1, 0.9, e))

# -------------------------------
# 4) FUNCIÓN WRAPPER QUE EJECUTA UN EXPERIMENTO
# -------------------------------
def run_config(args):
    """
    Ejecuta una configuración de hiperparámetros
    Se ejecuta en procesos paralelos
    """
    param, alpha, gamma, decay = args
    
    try:
        print(f"Ejecutando: {param}={alpha if param=='alpha' else gamma if param=='gamma' else decay}")
        
        # Entrenar con esta configuración
        reward = entrenar_configuracion_analisisSensibilidad(
            alpha, gamma, decay, num_episodios=5
        )
        
        resultado = {
            "parametro": param,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_decay": decay,
            "recompensa": reward,
            "status": "éxito"
        }
        
    except Exception as e:
        print(f" Error en {param}: {e}")
        resultado = {
            "parametro": param,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_decay": decay,
            "recompensa": None,
            "status": f"error: {str(e)}"
        }
    
    return resultado

# -------------------------------
# 5) PROGRAMA PRINCIPAL (WINDOWS)
# -------------------------------
if __name__ == "__main__":
    resultados = []

    # Ejecutar en paralelo (4 procesos)
    with Pool(processes=7) as pool:  # usa 4 cores (4 procesos en paralelo)
        for result in pool.imap_unordered(run_config, params):
            # pool.imap_unordered toma CADA elemento de params y se lo pasa a run_config
            # Esto sucede en paralelo para 4 elementos a la vez
            resultados.append(result)
            # guardamos parcial
            pd.DataFrame(resultados).to_csv("Result_tunnig_hiperparametros.csv", index=False)
            print("Guardado parcial:", result)

    print("\nAnálisis de Tunnig Hiperparametros finalizado. Resultados en Result_tunnig_hiperparametros.csv")