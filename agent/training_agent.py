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

# ========================================================
# 1. CONFIGURACIÓN DE PARAMETROS PARA EL ENTRENAMIENTO
# =======================================================
MODO = "real"  # "real" o "artificial"
NUM_EPISODIOS = 2
SHOVEL_NAMES = ['PH002', 'EX004', 'PH003', 'PH001', 'CF001', 'CF002']
TICK_JSON_REAL_PATH = r"C:\RL_model\agent\MINE-hudbay-2025-08-19.json"
LOG_CSV_PATH = f"log_entrenamiento_AgenteRl_{MODO}.csv"
QTABLE_PATH = f"q_table_{MODO}.pkl"

## COMENTARIO

# ============================================
# 2. CARGAR DATOS
# ============================================
if MODO  == "real":
    with open(TICK_JSON_REAL_PATH, "r") as f:
        tick_json = json.load(f)
    #total_ticks = len(tick_json)  #Estás contando todo el JSON, incluyendo "Elementos_Estat", "Parametros_Globales", etc.
    tick_keys = sorted([k for k in tick_json.keys() if k.isdigit()], key=int)
    total_ticks = len(tick_keys)
else:
    tick_json = generar_tick_json_artificial(SHOVEL_NAMES, num_ticks=10, num_trucks=5)  # 0 - 9
    with open("tick_json_artificial_guardado.json", "w") as f:
        json.dump(tick_json, f, indent=2)
    print("Datos artificiales guardados en tick_json_artificial_guardado.json")
    #total_ticks = len(tick_json)  #Estás contando todo el JSON, incluyendo "Elementos_Estat", "Parametros_Globales", etc.
    tick_keys = sorted([k for k in tick_json.keys() if k.isdigit()], key=int)
    total_ticks = len(tick_keys)

# ============================================
# 3. ENTRENAMIENTO DEL AGENTE RL
# ============================================
agent = QLearningAgent(actions=SHOVEL_NAMES)
rewards_por_episodio = []
logs_entrenamiento = []  # Para guardar trazabilidad completa

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

#==========Entrenamiento del Agente RL, por EPISODIOS ==============================================================
for ep in range(NUM_EPISODIOS):
    print(f"\n--- Episodio {ep + 1} ---")
    total_reward = 0  # Total de recompensas

    for tick_index in range(total_ticks):
        #tick_data = tick_json[str(tick_index)]
        tick_data = tick_json[tick_keys[tick_index]]  # "5", "13", "27", "41": Cualquier secuencia
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
            next_tick_index = (tick_index + 1) % total_ticks
            next_tick_data = tick_json[tick_keys[next_tick_index]]

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

            # Logs visibles por consola
            #print(f"Paso {paso} | Acción: {action} | Recompensa: {reward}")
            print(f"Tick {tick_index} | Camión: {truck_id} | Acción: {action} | Recompensa: {reward}")
            print(f" → ETA de la acción elegida: {truck_etas.get(action, 'N/A')}")
            print(" → Recompensas por cada Accion:", recompensas_acciones_posibles)
            print(f" → Mejor Accion {mejor_accion} | Decision_Optima {decision_optima}")
            print_q_vals(state, valid_actions, agent.q_table)
            print_detalle_acciones(valid_actions, shovels_info, truck_etas)

            logs_entrenamiento.append({
                "episodio": ep + 1,
                #"paso": paso,
                "tick": tick_index,
                "truck_id": truck_id,
                "estado": state,
                "accion_elegida": action,
                "recompensa": reward,
                "recompensas_posibles": recompensas_acciones_posibles,  # Recompensas de todas las acciones posibles
                "q_valores": {a: agent.q_table.get(state, {}).get(a, 0.0) for a in valid_actions},
                #============================================================================
                "mejor_accion": mejor_accion,
                "decision_optima": decision_optima,
                "pala_asignada": action,
                #===========================================================================
                "palas_info": {
                    a: {
                        "state": shovels_info.get(a, {}).get("state"),
                        "queue_count": shovels_info.get(a, {}).get("queue_count")
                    } for a in valid_actions
                },
                #===========================================================================
                "truck_etas": truck_etas,  # Agregar ETAs del camión para análisis
                "eta_accion_elegida": truck_etas.get(action, None)  # ETA de la acción elegida
            })


    agent.decay_epsilon()
    rewards_por_episodio.append(total_reward)
    print(f"Recompensa total del episodio {ep + 1}: {total_reward}")

# ============================================
# GUARDAR Q-TABLE
# ============================================
#filename_qtable = f"q_table_{modo}.pkl"
with open(QTABLE_PATH, "wb") as f:
    pickle.dump(agent.q_table, f)
print(f"Q-table guardada en {QTABLE_PATH}")

# =========================
# Mostrar Q-table
# =========================
print("\n--- Muestra de Q-table entrenada ---")
#for estado, acciones in list(agent.q_table.items())[:10]:
for estado, acciones in list(agent.q_table.items()):
    print(f"Estado: {estado} → Q-valores: {acciones}")

# ============================================
# GUARDAR LOG DEL ENTRENAMIENTO
# ============================================
#Opcion2:
df_log = pd.DataFrame(logs_entrenamiento)
df_log.to_csv(LOG_CSV_PATH, index=False)
print(f"Log de entrenamiento guardado en: {LOG_CSV_PATH}")

# =========================
# 5. Graficar recompensas
# =========================
plt.figure(figsize=(8, 4))
plt.plot(rewards_por_episodio,  marker='o')
plt.title(f"Recompensas por episodio ({MODO})")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"recompensas_{MODO}.png")
plt.show()