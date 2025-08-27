#training_agent.py
import random
from q_learning_agent import QLearningAgent
from state_builder import state_builder3
from rewards import calcular_recompensa
import pickle
import matplotlib.pyplot as plt
import json
from generar_data_artifitial import generar_tick_json_artificial
import pandas as pd
import os
import gc

# ========================================================
# 1. CONFIGURACIÓN DE PARAMETROS PARA EL ENTRENAMIENTO
# =======================================================
MODO = "real"  # "real" o "artificial"
NUM_EPISODIOS = 2
SHOVEL_NAMES = ['PH002', 'EX004', 'PH003', 'PH001', 'CF001', 'CF002']
TICK_JSON_REAL_PATH = r"C:\RL_model\agent\MINE-hudbay-2025-08-19.json"
LOG_CSV_PATH = f"log_entrenamiento_AgenteRl_{MODO}.csv"
QTABLE_PATH = f"q_table_{MODO}.pkl"


# Variables globales para datos pre-cargados
TICK_JSON = None
TICK_KEYS = None
TOTAL_TICKS = None

#===================================================
# 1. INICIALIZACIÓN (CARGAR DATOS UNA SOLA VEZ)
# ========================================================
def inicializar_datos():
    """Carga los datos UNA vez al inicio (optimización clave)"""
    global TICK_JSON, TICK_KEYS, TOTAL_TICKS
    
    if TICK_JSON is not None:
        return  # Ya está inicializado
    
    print("Cargando datos de entrenamiento...")
    
    if MODO == "real":
        import json
        with open(TICK_JSON_REAL_PATH, "r") as f:
            TICK_JSON = json.load(f)
        TICK_KEYS = sorted([k for k in TICK_JSON.keys() if k.isdigit()], key=int)
    else:
        from generar_data_artifitial import generar_tick_json_artificial
        TICK_JSON = generar_tick_json_artificial(SHOVEL_NAMES, num_ticks=10, num_trucks=5)
        TICK_KEYS = sorted([k for k in TICK_JSON.keys() if k.isdigit()], key=int)
    
    TOTAL_TICKS = len(TICK_KEYS)
    print(f"Datos cargados: {TOTAL_TICKS} ticks")


# ============================================
# 3. FUNCIONES AUXILIARES
# ============================================
def print_q_vals(state, valid_actions, q_table):
    print(f"  → Q-valores en el estado actual:")
    for a in valid_actions:
        q_val = q_table.get(state, {}).get(a, 0.0)
        print(f"    Acción: {a:<6} → Q: {q_val:.4f}")

def print_detalle_acciones(valid_actions, shovel_dict, truck_etas=None):
    print("  → Detalles de palas:")
    for pala in valid_actions:
        info = shovel_dict.get(pala, {})
        estado = info.get("state", "N/A")
        cola = info.get("queue_count", "N/A")
        eta = truck_etas.get(pala, "N/A") if truck_etas else "N/A"
        print(f"Pala: {pala:<6} | Estado: {estado:<2} | Cola: {cola:<2} | ETA: {eta}")


def entrenar_configuracion_analisisSensibilidad(alpha, gamma, epsilon_decay, num_episodios):
    # # Crear agente con la configuración pasada como parámetro, internamente para hacer el Tuning de Hiperparámetros
    """
    Entrena un agente con configuración específica - OPTIMIZADA para paralelismo
    Devuelve recompensa promedio de los episodios
    """
    
    # Inicializar datos si no están cargados
    if TICK_JSON is None:
        inicializar_datos()
    
    # Crear agente con la configuración específica
    agent = QLearningAgent(
        actions=SHOVEL_NAMES,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay
    )

    rewards_por_episodio = []

    #==========Entrenamiento del Agente RL, por EPISODIOS ==============================================================
    for ep in range(num_episodios):
        print(f"\n--- Episodio {ep + 1} ---")
        total_reward = 0  # Total de recompensas

        for tick_index in range(TOTAL_TICKS):
            #tick_data = tick_json[str(tick_index)]
            tick_data = TICK_JSON[TICK_KEYS[tick_index]]  # "5", "13", "27", "41": Cualquier secuencia
            truck_states = tick_data["truck_states"]
            shovels_info = tick_data["shovel_states"]

            for truck_id, truck_info in truck_states.items():
                tick_info = {
                    "truck_states": {truck_id: truck_info},
                    "shovel_states": shovels_info
                }

                # 1. Construir estado actual
                states, truck_names_list = state_builder3(tick_info, SHOVEL_NAMES)
                state = states[0]  # usamos solo el camion actual

                # 2. Acciones válidas
                valid_actions = SHOVEL_NAMES    # ← Ahora todas las acciones están disponibles

                if not valid_actions:
                    print(f"No hay acciones válidas para el camión {truck_id} en tick {tick_index}")
                    continue

                # 3. Elegir acción
                action = agent.choose_action(state, valid_actions)

                # 4. Obtener next_tick (mismo camión, siguiente tick)
                next_tick_index = (tick_index + 1) % TOTAL_TICKS
                next_tick_data = TICK_JSON[TICK_KEYS[next_tick_index]]

                next_truck_info = next_tick_data["truck_states"].get(truck_id, truck_info)
                next_tick_info = {
                    "truck_states": {truck_id: next_truck_info},
                    "shovel_states": next_tick_data["shovel_states"]
                }
                next_states, next_truck_names_list = state_builder3(next_tick_info, SHOVEL_NAMES)
                next_state = next_states[0]

                # 5. Extraer ETAs del camión actual
                truck_etas = truck_info.get("ETA", {})
                
                # 5. Recompensa
                reward = calcular_recompensa(
                    status=next_truck_info.get("status", "waiting for shovel"), # valor por defecto, por si no se encuentra el status en next_truck_info
                    action=action,
                    shovels_info=shovels_info,
                    truck_etas=truck_etas  # Agregar ETAs del camión
                )

                # 5.1 Calcular recompensas para todas las acciones posibles (debug, análisis, logging)
                recompensas_acciones_posibles = {
                    a: calcular_recompensa(
                        status=next_truck_info.get("status", "waiting for shovel"),
                        action=a,
                        shovels_info=shovels_info,
                        truck_etas=truck_etas  # Agregar ETAs del camión
                    ) for a in valid_actions
                }

                # ==========Propuesta de Metricas ==================================================
                mejor_accion = max(recompensas_acciones_posibles, key=recompensas_acciones_posibles.get)
                decision_optima = (action == mejor_accion)
                #waiting_time = truck_info.get("waiting_time", 0)


                # 6. Validación de acciones para próximo estado
                next_valid_actions = valid_actions

                # 7. Actualizar agente
                agent.update(state, action, reward, next_state, next_valid_actions)

                total_reward += reward


        agent.decay_epsilon()
        rewards_por_episodio.append(total_reward)
        print(f"Episodio {ep+1}: Recompensa {total_reward}")
        # Liberar memoria después de cada episodio
        #gc.collect()

    # Devuelve la recompensa media (para comparaciones en sensibilidad)
    return sum(rewards_por_episodio) / len(rewards_por_episodio)