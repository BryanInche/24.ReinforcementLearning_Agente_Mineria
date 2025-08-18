import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
__file__	Ruta completa del archivo actual (Probar_Agente_NewData.py)
os.path.dirname(__file__)	Carpeta donde está el script: .../RL_model/test
os.path.join(..., '..')	Subimos un nivel: .../RL_model/
os.path.abspath(...)	Convertimos la ruta a absoluta para evitar ambigüedad
sys.path.append(...)	Añadimos esa ruta (RL_model) al sistema de búsqueda de módulos
"""
# Carga de módulos del agente
import pickle
from agent.q_learning_agent import QLearningAgent  # asegúrate de que esta clase esté disponible
from agent.state_builder import state_builder3
from agent.rewards import calcular_recompensa


# ---1. Simulamos un nuevo tick (observación del entorno)-----------------
tick_info_nuevo = {
    "truck_states": {
        "CM04": {
            "name": "CM04",
            "status": "unloading",  # → mapeado a 0 'waiting for shovel': 0,'moving load': 1, 'moving unload': 2,'waiting for dumper': 3,'loading': 4,'unloading': 5
            # Para "reconstruir una coordenada real"
            'position': [410 * 500 + 1, 16786 * 500 + 1],  # → produce bins (404, 16788)
            'tank_fuel_level': 60  # → bin alto (2)  if fuel is None: fuel < 20 -> return 0  # bajo; fuel < 50 -> return 1  # medio; return 2  # alto
        }
    },
    "shovel_states": {
        "PH002": {"state": 1, "queue_count": 1},
        "EX004": {"state": 1, "queue_count": 0},
        "PH003": {"state": 1, "queue_count": 0},
        "PH001": {"state": 4, "queue_count": 0},
        "CF001": {"state": 4, "queue_count": 0},
        "CF002": {"state": 3, "queue_count": 1}
    }
}

# Lista de palas disponibles
shovel_names = ['PH002', 'EX004' ,'PH003', 'PH001' ,'CF001' ,'CF002']

# ---2. Crear una nueva instancia del agente--------
# Aún no contiene la Q-table entrenada
agente_cargado = QLearningAgent(actions=shovel_names)  # Solo creas un objeto vacío

# ---3. Cargar la tabla Q entrenada desde archivo pickle------------------------------
with open(r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\q_table_artificial.pkl", "rb") as f:
    agente_cargado.q_table = pickle.load(f)  # Cargas la tabla previamente entrenada

# ---4. Mostrar la Q-table para confirmar que se ha cargado correctamente---------------------
print("Q-table cargada:")
#for estado, acciones in list(agente_cargado.q_table.items())[:5]:
for estado, acciones in list(agente_cargado.q_table.items()):
    print(f"{estado} → {acciones}")


# ---4. Construir el nuevo estado a partir del tick (estado del entorno)------------------------------------------
# Supón que tienes `tick_info_nuevo`
estado_nuevo = state_builder3(tick_info_nuevo, shovel_names)[0]
print("\n" + "="*80)
print(f"\nEstado actual construido por `state_builder3`:\n{estado_nuevo}")
print("="*80)

# 4.2 Verificar si ese estado ya estaba en la Q-table entrenada
if estado_nuevo in agente_cargado.q_table:
    print("El estado ya existe en la Q-table entrenada.")
    print(f"   → Q-valores: {agente_cargado.q_table[estado_nuevo]}")
else:
    print("El estado NO EXISTE en la Q-table entrenada.")
    print("   → Se inicializa automáticamente con Q=0.0 para cada acción.")

tick_actual = estado_nuevo[0]

# Mostrar estados con mismo tick para comparación
print(f"\n Buscando estados con tick {tick_actual} en la Q-table:")
for estado in agente_cargado.q_table:
    if estado[0] == tick_actual:
        print("Estado parecido:", estado)


# ---5. Elegir la acción recomendada por el agente entrenado-----------------------------------------
acciones_posibles = shovel_names  # o filtra si quieres
accion = agente_cargado.choose_action(estado_nuevo, acciones_posibles)
print("\n" + "="*80)
print(f"Acción recomendada por el agente entrenado: {accion}")
print("="*80)

# ---6. Calcular recompensa real (usando función personalizada)
recompensa = calcular_recompensa(
    tick_info_nuevo["truck_states"]["CM04"]["status"],
    accion,
    tick_info_nuevo["shovel_states"]
)
print(f"Recompensa obtenida: {recompensa:.2f}")

## --- 6.1 Recompensas inmediatas por acción posible en este estado ---------
print("\n→ Recompensas inmediatas por acción posible en este estado:")
for acc in acciones_posibles:
    r = calcular_recompensa(
        tick_info_nuevo["truck_states"]["CM04"]["status"],
        acc,
        tick_info_nuevo["shovel_states"]
    )
    print(f"  Acción: {acc:<6} → Recompensa estimada: {r:.4f}")


# ---11. Mostrar resumen de palas-----------------------------------------------------------------------
print("\n→ Detalle de palas en el entorno:")
for pala in acciones_posibles:
    info = tick_info_nuevo["shovel_states"].get(pala, {})
    print(f"  Pala: {pala:<6} | Estado: {info.get('state', 'N/A')} | Cola: {info.get('queue_count', 'N/A')}")



###################################################################################
len_antes = len(agente_cargado.q_table)
estado_existe_antes = estado_nuevo in agente_cargado.q_table
##################################################################################

# ---7. Simular siguiente estado (por ahora igual al actual)----------------------------------------------
# En producción lo ideal es capturar el nuevo estado del tick siguiente
siguiente_estado = estado_nuevo

# ---8. Aprendizaje online: actualizar Q-table------------------------------------------------
agente_cargado.update(estado_nuevo, accion, recompensa, siguiente_estado, acciones_posibles)

# ---9. Mostrar Q-valores del estado actualizado------------------------------------------------------
print("\n→ Matriz Q-valores para el estado actual tras actualización:")
q_vals_estado = agente_cargado.q_table.get(estado_nuevo, {})
for acc in acciones_posibles:
    print(f"  Acción: {acc:<6} → Q: {q_vals_estado.get(acc, 0.0):.4f}")

# -- Confirmar si el estado fue agregado
estado_existe_despues = estado_nuevo in agente_cargado.q_table
if not estado_existe_antes and estado_existe_despues:
    print(f"\n Nuevo estado AGREGADO correctamente a la Q-table.")


# -- 10. Guardar la Q-table actualizada------------------------------------------------------
ruta_salida = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\q_table_artificial_actualizada.pkl"
with open(ruta_salida, "wb") as f:
    pickle.dump(agente_cargado.q_table, f)
print(f"\n Q-table actualizada y guardada en:\n{ruta_salida}")