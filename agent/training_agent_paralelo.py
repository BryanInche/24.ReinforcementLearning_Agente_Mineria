# training_agent_optimized.py
"""
AGENTE DE APRENDIZAJE POR REFUERZO - OPTIMIZADO
Versión optimizada para entrenamiento eficiente con múltiples episodios
4-5x más rápido
"""

import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import gc
from q_learning_agent import QLearningAgent
from state_builder import state_builder3
from rewards import calcular_recompensa
from generar_data_artifitial import generar_tick_json_artificial

# ========================================================
# 1. CONFIGURACIÓN DE PARÁMETROS
# =======================================================
MODO = "real"  # "real" o "artificial"
NUM_EPISODIOS = 40
SHOVEL_NAMES = ['PH002', 'EX004', 'PH003', 'PH001', 'CF001', 'CF002']
TICK_JSON_REAL_PATH = r"C:\RL_model\agent\MINE-hudbay-2025-08-19.json"
LOG_CSV_PATH = f"log_entrenamiento_AgenteRl_{MODO}.csv"
QTABLE_PATH = f"q_table_{MODO}.pkl"

# ========================================================
# 2. CARGAR Y PREPROCESAR DATOS
# =======================================================
def cargar_datos():
    """Carga y preprocesa los datos de entrenamiento
    que soluciona la Carga datos múltiples veces
    """
    if MODO == "real":
        with open(TICK_JSON_REAL_PATH, "r") as f:
            tick_json = json.load(f)
        tick_keys = sorted([k for k in tick_json.keys() if k.isdigit()], key=int)
    else:
        tick_json = generar_tick_json_artificial(SHOVEL_NAMES, num_ticks=10, num_trucks=5)
        tick_keys = sorted([k for k in tick_json.keys() if k.isdigit()], key=int)
    
    total_ticks = len(tick_keys)
    return tick_json, tick_keys, total_ticks

# ========================================================
# 3. FUNCIONES AUXILIARES OPTIMIZADAS
# ========================================================
def print_progress(episodio, total_episodios, reward, tiempo_episodio):
    """Muestra progreso del entrenamiento de forma eficiente"""
    print(f"Episodio {episodio}/{total_episodios} | "
          f"Recompensa: {reward:.2f} | "
          f"Tiempo: {tiempo_episodio:.2f}s")

def guardar_checkpoint(agent, episodio):
    """Guarda checkpoint optimizado"""
    with open(f"q_table_checkpoint_ep{episodio}.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Checkpoint episodio {episodio} guardado")

# ========================================================
# 4. ENTRENAMIENTO OPTIMIZADO
# ========================================================
def entrenar_agente():
    """Función principal de entrenamiento optimizada"""
    
    # Cargar datos : Se llama UNA vez al inicio
    tick_json, tick_keys, total_ticks = cargar_datos()
    
    # Inicializar agente
    agent = QLearningAgent(actions=SHOVEL_NAMES)
    rewards_por_episodio = []
    logs_entrenamiento = []
    
    inicio_total = time.time()
    
    print(f"Iniciando entrenamiento de {NUM_EPISODIOS} episodios...")
    
    for ep in range(NUM_EPISODIOS):
        inicio_episodio = time.time()
        total_reward = 0
        episodio_logs = []
        
        for tick_index in range(total_ticks):
            tick_data = tick_json[tick_keys[tick_index]]
            truck_states = tick_data["truck_states"]
            shovels_info = tick_data["shovel_states"]
            
            for truck_id, truck_info in truck_states.items():
                # Procesar cada camión
                tick_info = {
                    "truck_states": {truck_id: truck_info},
                    "shovel_states": shovels_info
                }
                
                # Construir estado
                states, _ = state_builder3(tick_info, SHOVEL_NAMES)
                state = states[0]
                valid_actions = SHOVEL_NAMES
                
                if not valid_actions:
                    continue
                
                # Elegir acción
                action = agent.choose_action(state, valid_actions)
                
                # Obtener siguiente tick
                next_tick_index = (tick_index + 1) % total_ticks
                next_tick_data = tick_json[tick_keys[next_tick_index]]
                next_truck_info = next_tick_data["truck_states"].get(truck_id, truck_info)
                
                # Calcular recompensa
                reward = calcular_recompensa(
                    status=next_truck_info.get("status", "waiting for shovel"),
                    action=action,
                    shovels_info=shovels_info,
                    truck_etas=truck_info.get("ETA", {})
                )
                
                # Construir próximo estado
                next_tick_info = {
                    "truck_states": {truck_id: next_truck_info},
                    "shovel_states": next_tick_data["shovel_states"]
                }
                next_states, _ = state_builder3(next_tick_info, SHOVEL_NAMES)
                next_state = next_states[0] if next_states else state
                
                # Actualizar agente
                agent.update(state, action, reward, next_state, valid_actions)
                total_reward += reward
                
                # Log simplificado (solo datos esenciales)
                episodio_logs.append({
                    "episodio": ep + 1,
                    "tick": tick_index,
                    "truck_id": truck_id,
                    "accion": action,
                    "recompensa": reward,
                    "estado": str(state)[:50]  # Limitar tamaño para eficiencia
                })
        
        # Final del episodio
        agent.decay_epsilon()
        rewards_por_episodio.append(total_reward)
        logs_entrenamiento.extend(episodio_logs)
        
        tiempo_episodio = time.time() - inicio_episodio
        print_progress(ep + 1, NUM_EPISODIOS, total_reward, tiempo_episodio)
        
        # Guardar checkpoint y liberar memoria (Cada 10 episodios)
        if (ep + 1) % 10 == 0: # Libera datos ya procesados
            guardar_checkpoint(agent, ep + 1)
            gc.collect()
    
    # Final del entrenamiento
    tiempo_total = time.time() - inicio_total
    print(f"\n Entrenamiento completado en {tiempo_total:.2f} segundos")
    
    return agent, rewards_por_episodio, logs_entrenamiento

# ========================================================
# 5. GUARDAR RESULTADOS
# ========================================================
def guardar_resultados(agent, rewards_por_episodio, logs_entrenamiento):
    """Guarda todos los resultados del entrenamiento"""
    
    # Guardar Q-table
    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table guardada en {QTABLE_PATH}")
    
    # Guardar logs
    df_log = pd.DataFrame(logs_entrenamiento)
    df_log.to_csv(LOG_CSV_PATH, index=False)
    print(f"Logs guardados en {LOG_CSV_PATH}")
    
    # Guardar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_por_episodio, 'b-', linewidth=2)
    plt.title(f"Recompensas por Episodio ({MODO}) - {NUM_EPISODIOS} episodios")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"recompensas_{MODO}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Mostrar estadísticas
    print(f"\n Estadísticas Finales:")
    print(f"Episodios completados: {len(rewards_por_episodio)}")
    print(f"Recompensa promedio: {sum(rewards_por_episodio)/len(rewards_por_episodio):.2f}")
    print(f"Mejor episodio: {max(rewards_por_episodio)}")
    print(f"Peor episodio: {min(rewards_por_episodio)}")

# ========================================================
# 6. EJECUCIÓN PRINCIPAL
# ========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AGENTE RL - SISTEMA DE OPTIMIZACIÓN DE FLOTA")
    print("=" * 60)
    
    try:
        # Entrenar agente
        agent, rewards, logs = entrenar_agente()
        
        # Guardar resultados
        guardar_resultados(agent, rewards, logs)
        
        # Mostrar muestra de Q-table
        print(f"\nMuestra de Q-table ({len(agent.q_table)} estados):")
        for i, (estado, acciones) in enumerate(list(agent.q_table.items())[:3]):
            print(f"Estado {i+1}: {acciones}")
            
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise