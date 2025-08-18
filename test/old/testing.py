import sys, os
sys.path.append(os.path.abspath(".."))  # Añade el directorio padre al path para importar módulos personalizados

# Función que convierte la información del entorno a un estado RL
from agent.state_builder import state_builder2
import json


# ----1. Datos Artificiales -------------------------------
"""
truck_info = {
    "current_locacion": 66920,
    "status": "waiting for shovel",
    "tank_fuel_level": 45,
    #"load_blocked": 0
}
"""


"""
shovels_info = {
    "PH001": {"priority": 1, "coverage": 85, "enabled": True,  "main_state": 1},
    "PH002": {"priority": 1, "coverage": 70, "enabled": True,  "main_state": 1},
    "PH003": {"priority": 3, "coverage": 90, "enabled": True,  "main_state": 2},
    "PH004": {"priority": 2, "coverage": 60, "enabled": False, "main_state": 1}
}

# Orden fijo de palas (coherente entre ticks)
shovel_names = ["PH001", "PH002", "PH003", "PH004"]
"""

# ---- 2. Datos Cargados del Mine_Config y Tick_json --------------------------------

# 2.1 --- # --- Cargar archivo mine_config.json (estructura base de la mina, camiones y palas) --------
# y carga su contenido como un diccionario
with open(r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\config\mine_config.json") as f:
    config_data_base = json.load(f)


# -2.2 Extraer la info de shovels directamente desde el JSON ---
shovels_info = {}
for loadsite_id, loadsite_data in config_data_base["load_sites"].items():
    shovels = loadsite_data.get("shovels", {})  # accede a la sección de palas
    
    # Recorremos los sitios de carga (load_sites) del JSON, y dentro de cada uno, buscamos las palas.
    for shovel_id, shovel_data in shovels.items():
        shovel_name = shovel_data.get("name")
        if shovel_name:  # solo si el nombre existe
            # Se construye un diccionario shovels_info para cada pala, guardando su prioridad, cobertura, si está habilitada y su estado principal
            shovels_info[shovel_name] = {
                "priority": shovel_data.get("priority", -1),
                "coverage": shovel_data.get("coverage", -1),
                "enabled": shovel_data.get("enabled", False),
                "main_state": shovel_data.get("main_state", -1)
            }
# -------------------------------------------------------------------------------
# -2.3 Ordenar las palas por prioridad o por nombre (según se desee) ---
# Extraemos una lista ordenada de nombres de palas para mantener un orden fijo.
shovel_names = list(shovels_info.keys())


####### ---------------------------------------------------------- #########
# --2.4 Cargar el archivo de ticks (estado dinámico del sistema) ---
with open(r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\config\ticks.json") as f:
    tick_data = json.load(f)

# ---- 3.1 Seleccionamos el tick actual y el nombre del camión con el que deseamos trabajar
tick_actual = "0"
truck_id = "CM20"

#--- 2.5 Validamos que el camión efectivamente existe en ese tick. Esto da su estado dinámico, como "status"
truck_data_tick = tick_data[tick_actual]["truck_states"].get(truck_id)
if not truck_data_tick:
    raise ValueError(f"No se encontró el camión {truck_id} en el tick {tick_actual}")

# --2.6 Buscamos el camión en la estructura base para obtener sus datos estáticos como current_locacion, tank_fuel_level, etc ------------------
truck_data_config = None
for truck in config_data_base["truck"].values():
    if truck.get("name") == truck_id:
        truck_data_config = truck
        break

if not truck_data_config:
    raise ValueError(f"No se encontró el camión {truck_id} en mine_config.json")

    
# --- 2.7 Construcción final del estado de entrada del camión
# Se construye el diccionario truck_info combinando información dinámica del tick (status)
#  con información estática del config (posición, nivel de combustible).
##
# 🔹 config_data_base (mine_config.json): para obtener current_locacion y tank_fuel_level (datos estáticos)
# 🔹 tick_data (ticks.json): para obtener status (dato dinámico del tick actual)
# Esto asegura que cada campo proviene de la fuente correcta:
#     - current_locacion: estático, viene de mine_config.json
#     - status: dinámico, viene de ticks.json
#     - tank_fuel_level: estático, viene de mine_config.json

truck_info = {
    "current_locacion": truck_data_config.get("current_locacion", -1),
    "status": truck_data_tick.get("status", "unknown"),
     "tank_fuel_level": truck_data_config.get("tank_fuel_level", 45)  # simulado, podrías traerlo de otro lado si lo tienes
    #"load_blocked": truck_data_config.get("load_blocked", 0) 
}
# --------------------------------------------------------------


# ---3.  Generar el estado ------------------
# El orden es fijo y conocido, Este orden se mantiene constante, así que el agente debe tener el mismo orden de acciones válidas:
state = state_builder2(truck_info, shovels_info, shovel_names)

# ---3.1 Etiquetas legibles para visualizar ------------------
field_names = ["current_locacion", "status (discreto)", "fuel (discretizado)"]
field_names += [f"{name}_priority" for name in shovel_names]
field_names += [f"{name}_coverage" for name in shovel_names]
field_names += [f"{name}_main_state" for name in shovel_names]

# --- Imprimir estado generado de forma ordenada ---
print("Estado generado (campo: valor):\n" + "-"*40)
for name, value in zip(field_names, state):
    print(f"{name:25}: {value}")

#print("Estado generado:", state)

# 4. Filtramos las Acciones válidas para el agente en ese estado: palas con main_state == 1
valid_actions = [name for name in shovel_names if shovels_info[name]["main_state"] == 1]
print("Acciones válidas:", valid_actions)


"""
Ejemplo correcto en testing.py:
# Obtenido desde mine_config.json
truck_data_config = ...  # contiene current_locacion, fuel, etc.

# Obtenido desde ticks.json
truck_data_tick = ...  # contiene el status actual del camión

truck_info = {
    "current_locacion": truck_data_config.get("current_locacion", -1),
    "status": truck_data_tick.get("status", "unknown"),
    "tank_fuel_level": truck_data_config.get("tank_fuel_level", 45)
}
-- Con eso, no habrá ninguna confusión, porque estás construyendo truck_info de forma explícita y controlada antes de pasarlo al state_builder2.
"""


