import os
import sys
import json
import time
import pickle
import logging
from datetime import datetime
import pandas as pd
import yaml

# Importar funciones y clases del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.q_learning_agent import QLearningAgent
from agent.state_builder import state_builder3
from agent.rewards import calcular_recompensa

# Cargar archivo YAML
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Paths
QTABLE_PATH = config["paths"]["qtable_path"]
TICK_FILE = config["paths"]["tick_file"]
DECISION_LOG_PATH = config["paths"]["decision_log_path"]
SHOVEL_NAMES = config["agent"]["shovel_names"]


LOG_DIR = config["paths"]["log_dir"]
# Crear carpeta si no existe
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = config["logging"]["log_level"] # log_level indica cuán detallados serán los mensajes que se mostrarán o guardarán en el log
LOG_FILE = f"agente_rl_tick_unico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] RLDispatcher: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8') #,
        #logging.StreamHandler()
    ]
)

#===== Funciones Auxiliares =========

def get_decision_reason(state, action, q_table, exploracion, truck_etas=None, shovels_info=None):
    """
    Explica por qué se tomó una acción.
    Esta versión NO genera ETAs simuladas: usa sólo truck_etas si están presentes.
    """
    # Si el estado es totalmente nuevo -> explicación simple
    if state not in q_table:
        return "Estado nuevo, acción tomada por exploración"

    q_values = q_table.get(state, {})
    max_q = max(q_values.values()) if q_values else None

    if exploracion:
        base_reason = "Exploración" # ε-greedy (acción aleatoria)"
    elif max_q is not None and q_values.get(action) == max_q:
        base_reason = "Acción con mayor Q-valor conocida"
    else:
        base_reason = "Acción no óptima (Q subóptimo seleccionado)"

    # Información de ETA (si viene en el tick)
    eta_info = ""
    if truck_etas and isinstance(truck_etas, dict) and action in truck_etas:
        eta_to_action = truck_etas[action]
        if eta_to_action is not None:
            try:
                eta_info = f" | ETA: {float(eta_to_action):.1f} min"
            except Exception:
                eta_info = f" | ETA: {eta_to_action}"

            # comparar con otras palas (solo las que estén en truck_etas y activas si shovels_info disponible)
            if shovels_info:
                available_etas = {
                    s: truck_etas[s] for s in truck_etas
                    if s in shovels_info and truck_etas.get(s) is not None
                }
                if available_etas:
                    try:
                        min_eta = min(v for v in available_etas.values() if v is not None)
                        if eta_to_action == min_eta:
                            eta_info += " (MEJOR ETA)"
                        elif eta_to_action <= min_eta * 1.2:
                            eta_info += f" (cerca del óptimo: {min_eta:.1f})"
                        else:
                            eta_info += f" (no óptimo: mejor={min_eta:.1f})"
                    except Exception:
                        pass

    # Información de la pala elegida (estado, cola) si está en shovels_info
    shovel_info = ""
    if shovels_info and action in shovels_info:
        sd = shovels_info[action]
        
        #state_pala = sd.get("state", "N/A")
        state_pala = sd.get("main_state", "N/A")

        cola_pala = sd.get("queue_count", "N/A")
        if state_pala == 1:
            shovel_info = f" | Pala activa, cola: {cola_pala}"
        else:
            shovel_info = f" | Pala inactiva (estado: {state_pala})"

    return base_reason + eta_info + shovel_info #+ etas_info
#=====================================================================================================================

# ==== CARGAR AGENTE ====
agent = QLearningAgent(actions=SHOVEL_NAMES)
if os.path.exists(QTABLE_PATH):
    with open(QTABLE_PATH, "rb") as f:
        agent.q_table = pickle.load(f)
    logging.info(f"Q-table cargada con {len(agent.q_table)} estados conocidos.")
else:
    logging.warning("No se encontró Q-table previa. Iniciando desde cero.")

# ==== CARGAR TICK ACTUAL ====
logging.info(f"Cargando tick desde: {TICK_FILE}")
with open(TICK_FILE, "r") as f:
    tick_data = json.load(f)

truck_states = tick_data["truck_states"]
shovels_info = tick_data["shovel_states"]

# ==== CLASIFICAR CAMIONES ====
camiones_optimizados = []
camiones_fijos = []
for truck_id, truck_info in truck_states.items():
    if truck_info.get("status") == "waiting for shovel":
        camiones_optimizados.append(truck_id)
    else:
        camiones_fijos.append(truck_id)

logging.info(f"[Tick] Optimizar: {camiones_optimizados} | Fijos: {camiones_fijos}")


# ==== ACTUALIZAR Q-TABLE CON DECISIONES PENDIENTES (del tick anterior) ====
if os.path.exists(DECISION_LOG_PATH):
    with open(DECISION_LOG_PATH, "rb") as f:
        decisiones_previas = pickle.load(f)

    if decisiones_previas:
        logging.info(f"Procesando {len(decisiones_previas)} decisiones pendientes para actualización...")
        for dec in decisiones_previas:
            try:
                # Estado y acción del tick anterior
                state_prev = dec["state"]
                action_prev = dec["action"]
                valid_actions_prev = dec["valid_actions"]
                truck_id = dec["truck_id"]

                # Recompensa usando la situación actual del tick
                recompensa = calcular_recompensa(
                    status=truck_states.get(truck_id, {}).get("status", "N/A"),
                    action=action_prev,
                    shovels_info=shovels_info,
                    truck_etas=truck_states.get(truck_id, {}).get("ETA", {})
                )

                # Construimos el next_state a partir del tick actual
                tick_info_actual = {
                    "truck_states": {truck_id: truck_states[truck_id]},
                    "shovel_states": shovels_info
                }
                states_result_actual, _ = state_builder3(tick_info_actual, SHOVEL_NAMES)
                next_state = states_result_actual[0]

                # Si es nuevo estado, inicializamos sus valores Q
                if next_state not in agent.q_table:
                    agent.q_table[next_state] = {a: 0.0 for a in SHOVEL_NAMES}

                # Actualizamos Q-table
                agent.update(state_prev, action_prev, recompensa, next_state, valid_actions_prev)

            except Exception as e:
                logging.error(f"Error actualizando Q para camión {dec.get('truck_id')}: {e}")

        # Guardamos la Q-table actualizada
        with open(QTABLE_PATH, "wb") as f:
            pickle.dump(agent.q_table, f)
        logging.info("Q-table actualizada y guardada.")

    # Eliminamos las decisiones ya procesadas
    os.remove(DECISION_LOG_PATH)


# ==== PROCESAR TICK ACTUAL Y ELEGIR ACCIONES ====
resultados = []
decisiones_guardar = []

for truck_id in camiones_optimizados:
    try:
        tick_info = {
            "truck_states": {truck_id: truck_states[truck_id]},
            "shovel_states": shovels_info
        }

        # Construimos el estado actual
        states_result, _ = state_builder3(tick_info, SHOVEL_NAMES)
        state = states_result[0]

        # Si el estado no está en la Q-table, lo creamos con Q=0
        if state not in agent.q_table:
            agent.q_table[state] = {a: 0.0 for a in SHOVEL_NAMES}
            logging.warning(f"Estado nuevo agregado: {state}")


        truck_etas = truck_states[truck_id].get("ETA", {})
        # Las acciones validas serian solo las PALAS que estan disponibles en ETAS
        valid_actions = list(truck_etas.keys())
        if not valid_actions:
            continue
        
         # Elegir acción (exploración o explotación)
        action = agent.choose_action(state, valid_actions)
        exploracion = getattr(agent, "last_action_was_random", False)

        # Razón de la decisión (para logs)
        razon = get_decision_reason(state, action, agent.q_table, exploracion, truck_etas, shovels_info)

        # Guardar esta decisión para actualizarla en el próximo tick
        decisiones_guardar.append({
            "tick_id": tick_data.get("tick_id", None),
            "truck_id": truck_id,
            "state": state,
            "action": action,
            "valid_actions": valid_actions,
            "timestamp": datetime.now().isoformat()
        })

        # Agregar al resultado
        resultados.append({
            "Truck": truck_id,
            "Shovel": action,
            "Slot": truck_states[truck_id].get("slot", "NONE"),
            "Status": truck_states[truck_id].get("status", "NONE"),
            "Cost": truck_etas.get(action, 0.0),
            "Current": truck_states[truck_id].get("current_shovel", "NONE"),
            "ETA_reasignacion": truck_etas.get(action, 0.0),
            "ETA_Current": truck_etas.get(truck_states[truck_id].get("current_shovel", "NONE"), 0.0),
            "Changed": action != truck_states[truck_id].get("current_shovel", "NONE")
        })

        logging.info(f"Camión {truck_id} Asignar a {action} | Razón: {razon}")

    except Exception as e:
        logging.error(f"Error procesando camión {truck_id}: {e}")

# ==== GUARDAR DECISIONES PARA EL PRÓXIMO TICK ====
with open(DECISION_LOG_PATH, "wb") as f:
    pickle.dump(decisiones_guardar, f)
logging.info(f"Guardadas {len(decisiones_guardar)} decisiones para reentrenamiento en próximo tick.")

# ==== MOSTRAR RESULTADOS ====
df_result = pd.DataFrame(resultados)
print(df_result)
