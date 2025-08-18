#validation_agent_production
#=======================================================================
# 1. Importación y configuración de entorno
#=======================================================================
import os, sys
# Agregamos el directorio padre al path para poder importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pickle
import json
from agent.q_learning_agent import QLearningAgent
from agent.state_builder import state_builder3
from agent.rewards import calcular_recompensa

import logging
from datetime import datetime

# ================================================================
# 2. Configuración de parámetros y rutas
# ================================================================
# Agente RL entrenado (Matriz Q)
QTABLE_PATH = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\q_table_real.pkl"

# Cada cuántos ticks se guarda la Q-table en produccion 
QTABLE_SAVE_INTERVAL = 5  # cada cuántos ticks guardamos

# Acciones posibles: nombres de palas que se pueden asignar
SHOVEL_NAMES = ['PH002', 'EX004', 'PH003', 'PH001', 'CF001', 'CF002']

# Datos a Procesar(Inputs) en formato json
TICK_FILE = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\MINE-hudbay-TicksTrainRL-TIME-2025-08-01.json"

#==========================================================================
# 3. Configuración del Logger
#========================================================================
# Ruta donde se guardaran los LOGS de los Procesamientos Agente RL
LOG_DIR = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\logs"
LOG_FILE = f"agente_rl_produccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)
# Asegurarse que el directorio exista
os.makedirs(LOG_DIR, exist_ok=True) # Crear carpeta si no existe
# Configuración del logger
logging.basicConfig(
    level=logging.INFO,  # Esto asegura que todo lo importante se guarda tanto en consola como en archivo Log
    format="%(asctime)s [%(levelname)s] RLDispatcher: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

#======================================
# 4. Funciones auxiliares
#=========================================
## 4.1 La función revisa si el valor anterior era “Nuevo estado” o si hubo una diferencia relevante, y lo imprime para seguimiento.
def imprimir_actualizacion(state, action, valor_anterior, valor_nuevo, tick_id, truck_id):
    """Imprime información detallada sobre la actualización de Q-values"""
    if isinstance(valor_anterior, str) and valor_anterior == "Nuevo estado":
        print(f"\nNUEVO ESTADO en tick {tick_id} para camión {truck_id}")
        print(f"    Estado: {state}")
        print(f"    Acción: {action}")
        print(f"    Valor Q inicial: {round(valor_nuevo, 6)}")
    elif abs(valor_nuevo - valor_anterior) > 1e-6:
        print(f"\n  Q-VALUE ACTUALIZADO en tick {tick_id} para camión {truck_id}")
        print(f"    Estado: {state}")
        print(f"    Acción: {action}")
        print(f"    Valor anterior: {round(valor_anterior, 6)}")
        print(f"    Valor nuevo:    {round(valor_nuevo, 6)}")
        print(f"    Diferencia:     {round(valor_nuevo - valor_anterior, 6)}")


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
        state_pala = sd.get("state", "N/A")
        cola_pala = sd.get("queue_count", "N/A")
        if state_pala == 1:
            shovel_info = f" | Pala activa, cola: {cola_pala}"
        else:
            shovel_info = f" | Pala inactiva (estado: {state_pala})"

    return base_reason + eta_info + shovel_info #+ etas_info


def mostrar_comparacion_opciones(state, action, q_table, truck_etas, shovels_info, exploracion):
    """
    Muestra por consola una comparación entre las palas (usa solo truck_etas reales).
    Si truck_etas es None, mostrará ETA=N/A.
    """
    logging.info(f"\n Justificacion:")
    #logging.info(f"   Estado: {state}")
    logging.info(f"   Acción seleccionada: {action} ({'Exploración' if exploracion else 'Explotación'})")
    logging.info(f"   {'─'*80}")

    for pala in SHOVEL_NAMES:
        q_val = q_table.get(state, {}).get(pala, 0.0)
        shovel_data = shovels_info.get(pala, {})
        state_pala = shovel_data.get("state", "N/A")
        cola_pala = shovel_data.get("queue_count", 0)

        # ETA info desde truck_etas (si existe)
        eta_info = "N/A"
        if truck_etas and isinstance(truck_etas, dict) and pala in truck_etas:
            eta_val = truck_etas[pala]
            if eta_val is not None:
                try:
                    eta_info = f"{float(eta_val):.1f} min"
                except Exception:
                    eta_info = str(eta_val)

        estado_pala_str = "Activa" if state_pala == 1 else f"Inactiva ({state_pala})"
        selector = ">" if pala == action else " "
        logging.info(f"   {selector} {pala:6} | Q: {q_val:6.3f} | ETA: {eta_info:>8} | {estado_pala_str:15} | Cola: {cola_pala}")
        #print(f"   {selector} {pala:6} | Q: {q_val:6.3f} | ETA: {eta_info:>8} | {estado_pala_str:15} | Cola: {cola_pala}")

    #print(f"   {'─'*80}")
    logging.info(f"   {'─'*80}")

# ======================================
# 5. Carga del agente entrenado
# ======================================
agent = QLearningAgent(actions=SHOVEL_NAMES)

# Cargar Q-table entrenada: Si ya existe una Q-table guardada, la cargamos
if os.path.exists(QTABLE_PATH):
    with open(QTABLE_PATH, "rb") as f:
        agent.q_table = pickle.load(f)
    print(f"Q-table cargada exitosamente desde: {QTABLE_PATH}")
    print(f"   Estados conocidos: {len(agent.q_table)}")
else:
    print("No se encontró Q-table previa. Se iniciará desde cero.")

# Guardamos una copia de la Q-table inicial para comparar después
q_table_inicial = agent.q_table.copy()

# =====================================================
# 6. Cargar Datos De Producción (ticks REALES)
# =====================================================
print(f"\nCargando datos de producción desde: {TICK_FILE}")
with open(TICK_FILE, "r") as f:
    nuevos_ticks = json.load(f)

# Extraemos y ordenamos las claves (ticks numéricos) de menor a mayor
tick_keys = sorted([k for k in nuevos_ticks if k.isdigit()], key=int)
total_ticks = len(tick_keys)
print(f"Datos cargados: {total_ticks} ticks encontrados")

# ===============================================================
# 7. Flujo de Procesamiento del Agente RL en Producción
# ==============================================================
logging.info("Iniciando validación del agente en producción con datos REALES...")

# Métricas globales para toda la ejecución
metricas_globales = {
    "total_ticks_procesados": 0,
    "total_camiones_optimizados": 0,
    "total_camiones_fijos": 0,
    "total_camiones_error": 0,
    "recompensa_total_acumulada": 0.0,
    "estados_nuevos_encontrados": 0,
    "acciones_por_pala": {a: 0 for a in SHOVEL_NAMES},
    "decisiones_exploracion": 0,
    "decisiones_explotacion": 0,
    "q_values_actualizados": 0,
    "estados_unicos_vistos": set()
}

# ==========================================
# 7.1 Se procesan de a pares de ticks (actual y siguiente)
# ==========================================
for idx, tick_id in enumerate(tick_keys[:-1]):
    tick_actual = nuevos_ticks[tick_id] # tick actual
    tick_siguiente = nuevos_ticks[tick_keys[idx + 1]] # tick siguiente
    
    truck_states = tick_actual["truck_states"]
    shovels_info = tick_actual["shovel_states"]

    # ==============================================================================
    # 7.2 Clasificamos los camiones
    # ============================================================================
    camiones_optimizados = []
    camiones_fijos = []
    camiones_error = []

    # Logica para poner camiones Fijadas o Libres para Optimizar
    for truck_id, truck_info in truck_states.items():
        if truck_info.get("status") != "waiting for shovel":
            camiones_fijos.append(truck_id)
        else:
            camiones_optimizados.append(truck_id)

    # Actualizar métricas globales
    metricas_globales["total_ticks_procesados"] += 1
    metricas_globales["total_camiones_fijos"] += len(camiones_fijos)

    # ============================================================================
    # Logging inicial del tick
    # ============================================================================
    logging.info(f"[Tick {tick_id}] INICIO -> Optimizar: {camiones_optimizados} | Fijos: {camiones_fijos}")

    camiones_realmente_optimizados = []
    
    # 7.3 Por cada camión optimizable
    for truck_id in camiones_optimizados:
        try:
            # Extraer información del camión y palas para el estado actual
            tick_info = {
                "truck_states": {truck_id: truck_states[truck_id]},
                "shovel_states": shovels_info
            }

            # Construir estado actual
            states_result, truck_names_result = state_builder3(tick_info, SHOVEL_NAMES)
            state = states_result[0]  # Tomamos el estado del primer (y único) camión

            # Chequeo de existencia en la Q-table entrenada
            if state in agent.q_table:
                logging.info(f"Estado encontrado en Q-table: {state}")
            else:
                logging.warning(f"Estado Nuevo NO encontrado en Q-table: {state}")


            # Filtrar acciones válidas con ETAs reales
            truck_etas = truck_states[truck_id].get("ETA", {})
            valid_actions = list(truck_etas.keys())
            #valid_actions = SHOVEL_NAMES

            # Verificar si es un estado nuevo
            if state not in q_table_inicial:
                metricas_globales["estados_nuevos_encontrados"] += 1
                logging.info(f"NUEVO ESTADO detectado con respecto al entrenamiento: {state}")

            # Si la accion no es valida entonces ese Camion, se agrega como Error, porque no se procesara
            if not valid_actions:
                camiones_error.append(truck_id)
                continue

            # El agente decide una pala
            #=============================================================================================
            #El agente decide qué acción tomar (en este caso, a qué pala enviar el camión) basándose en el 
            #state actual del entorno y las acciones válidas (valid_actions, que son las palas disponibles 
            #en ese momento).
            #==============================================================================================
            action = agent.choose_action(state, valid_actions)

            #=========================================================================================
            # ¿Exploración o explotación?
            # Esta línea verifica si la acción que acaba de tomar fue por exploración (decisión aleatoria) 
            #o por explotación (usando la política aprendida). Esto depende si el agente guarda una propiedad
            # llamada last_action_was_random. Si no existe, asumimos False.
            #========================================================================================
            exploracion = getattr(agent, "last_action_was_random", False)
            
            # Actualizar métricas de exploración/explotación: Si fue una acción por exploración, sumamos 1 a ese contador.
            if exploracion:
                metricas_globales["decisiones_exploracion"] += 1
            else:
                metricas_globales["decisiones_explotacion"] += 1

            # =============================================================
            # Preparación info del siguiente estado en el siguiente tick
            # ============================================================
            next_truck_info = tick_siguiente["truck_states"].get(truck_id, truck_states[truck_id])
            next_tick_info = {
                "truck_states": {truck_id: next_truck_info},
                "shovel_states": tick_siguiente["shovel_states"]
            }

            # Construir siguiente estado
            next_states_result, _ = state_builder3(next_tick_info, SHOVEL_NAMES)
            # Tomamos el estado del primer (y único) camión
            next_state = next_states_result[0]
 
            # ==============================
            # Cálculo de recompensa
            # ==============================
            reward = calcular_recompensa(
                status=next_truck_info.get("status", "N/A"),
                action=action,
                shovels_info=shovels_info,
                truck_etas=truck_etas
            )

            # Obtener el valor Q anterior para este estado y acción.
            # Si el estado no existe aún en la Q-table, marcamos como "Nuevo estado".
            # agent.q_table es un diccionario con claves de estados.
            valor_anterior = agent.q_table.get(state, {}).get(action, "Nuevo estado")

            # El agente actualiza la Q-table
            agent.update(state, action, reward, next_state, valid_actions)

            # Valor Q después de la actualización
            valor_nuevo = agent.q_table[state][action]

            # Verificar si el Q-value cambió
            # Verifica que el valor anterior no sea un string (como "Nuevo estado").
            # Verifica que el nuevo valor Q sea diferente al anterior (más que un pequeño margen)
            if isinstance(valor_anterior, (int, float)) and abs(valor_nuevo - valor_anterior) > 1e-6: 
                metricas_globales["q_values_actualizados"] += 1

            # Justificación: Explicación de la decisión con información detallada
            razon = get_decision_reason(state, action, agent.q_table, exploracion, truck_etas, shovels_info)

            # ==============================
            # Logging detallado por camión
            # ==============================
            q_vals_actuales = agent.q_table[state]
            q_string = ", ".join([f"{a}: {round(v, 3)}" for a, v in q_vals_actuales.items()])

            logging.info(f"Camión {truck_id}: Estado → {state}")
            logging.info(f"  - Q-values: {q_string}")
            logging.info(f"  - Acción: {action} | Q: {round(valor_nuevo, 3)} | Recompensa: {round(reward, 3)} ") # | Explora: {exploracion}
            logging.info(f"  - Justificación: {razon}")

            # Add logging.info() de las comparaciones
            mostrar_comparacion_opciones(state, action, agent.q_table, truck_etas, shovels_info, exploracion)
            #coomparativas = mostrar_comparacion_opciones(state, action, agent.q_table, truck_etas, shovels_info, exploracion)
            #logging.info(coomparativas)

             # ===================================================
            # Logging detallado por pala seleccionada (Accion)
            # ===================================================
            shovel_data = shovels_info.get(action, {})
            estado_pala = shovel_data.get("state", "N/A")
            cola_en_pala = shovel_data.get("queue_count", "N/A")
            logging.info(f"  - Pala {action}: estado={estado_pala}, cola={cola_en_pala}")

            # ====================================================
            # Actualizar métricas globales
            # ====================================================
            metricas_globales["total_camiones_optimizados"] += 1
            metricas_globales["recompensa_total_acumulada"] += reward
            metricas_globales["acciones_por_pala"][action] += 1
            metricas_globales["estados_unicos_vistos"].add(state)
            camiones_realmente_optimizados.append(truck_id)


        except Exception as e:
            logging.error(f"Error al procesar camión {truck_id}: {e}")
            camiones_error.append(truck_id)

    # =======================================
    # Guardado periódico de la Q-table
    # =======================================
    if (idx + 1) % QTABLE_SAVE_INTERVAL == 0 or (idx + 1) == total_ticks:
        with open(QTABLE_PATH, "wb") as f:
            pickle.dump(agent.q_table, f)
        logging.info(f"[Tick {tick_id}] Q-table guardada exitosamente.")

# =======================================================
# ANÁLISIS FINAL Y MÉTRICAS DE PERFORMANCE
# =======================================================
print("\n" + "="*80)
print("MÉTRICAS FINALES DE VALIDACIÓN EN PRODUCCIÓN")
print("="*80)

# Análisis de estados nuevos
#estados_nuevos_agregados = analizar_estados_nuevos(q_table_inicial, agent.q_table)

# Métricas generales
print(f"\n MÉTRICAS GENERALES:")
print(f"  Total de ticks procesados:        {metricas_globales['total_ticks_procesados']}")
print(f"  Total de camiones optimizados:    {metricas_globales['total_camiones_optimizados']}")
print(f"  Total de camiones fijos:          {metricas_globales['total_camiones_fijos']}")
print(f"  Total de camiones con error:      {metricas_globales['total_camiones_error']}")
print(f"  Estados únicos vistos:            {len(metricas_globales['estados_unicos_vistos'])}")

# Métricas de aprendizaje
print(f"\n MÉTRICAS DE APRENDIZAJE:")
print(f"  Estados nuevos encontrados:       {metricas_globales['estados_nuevos_encontrados']}")
print(f"  Q-values actualizados:            {metricas_globales['q_values_actualizados']}")
print(f"  Decisiones por exploración:       {metricas_globales['decisiones_exploracion']}")
print(f"  Decisiones por explotación:       {metricas_globales['decisiones_explotacion']}")
if metricas_globales['total_camiones_optimizados'] > 0:
    tasa_exploracion = (metricas_globales['decisiones_exploracion'] / metricas_globales['total_camiones_optimizados']) * 100
    print(f"  Tasa de exploración:             {tasa_exploracion:.1f}%")

# Métricas de rendimiento
print(f"\n MÉTRICAS DE RENDIMIENTO:")
print(f"  Recompensa total acumulada:       {round(metricas_globales['recompensa_total_acumulada'], 2)}")
if metricas_globales['total_camiones_optimizados'] > 0:
    recompensa_promedio = metricas_globales['recompensa_total_acumulada'] / metricas_globales['total_camiones_optimizados']
    print(f"  Recompensa promedio por camión:   {round(recompensa_promedio, 3)}")

# Distribución de acciones por pala
print(f"\n DISTRIBUCIÓN DE ACCIONES POR PALA:")
for pala, count in metricas_globales['acciones_por_pala'].items():
    if count > 0:
        porcentaje = (count / metricas_globales['total_camiones_optimizados']) * 100
        print(f"  - {pala}: {count} veces ({porcentaje:.1f}%)")

# Resumen de la Q-table
print(f"\n RESUMEN DE LA Q-TABLE:")
print(f"  Estados iniciales:               {len(q_table_inicial)}")
print(f"  Estados finales:                 {len(agent.q_table)}")
print(f"  Estados agregados:               {len(agent.q_table) - len(q_table_inicial)}")

print("\n Proceso de validación en producción completado exitosamente!")
print(f" Logs guardados en: {LOG_PATH}")
print(f" Q-table actualizada guardada en: {QTABLE_PATH}")