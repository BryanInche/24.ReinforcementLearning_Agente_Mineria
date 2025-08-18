import json
from typing import List, Tuple
#-----------------------------------------------------------------------------------------------
# Paso 1: Convierte el nivel de combustible (fuel) en un valor discreto (entero: 0, 1 o 2) para usarlo como parte del estado.
def discretizar_fuel(fuel):
    if fuel is None:
        return 0
    if fuel < 20:
        return 0  # bajo
    elif fuel < 50:
        return 1  # medio
    else:
        return 2  # alto

# Paso 2: Mapea el texto del estado del camión (por ejemplo "waiting for shovel") a un número entero, para poder representarlo como un estado discreto.
status_mapping = {
    'waiting for shovel': 0,  ##
    'moving load': 1,    ## 
    'moving unload': 2,  ## 
    'waiting for dumper': 3, ##
    'loading': 4, ##
    'unloading': 5,
    'reverse': 6  # Agregado para el estado "reverse" que aparece en el JSON
}

# Paso 3: Discretizar posición X o Y
def discretizar_position(valor, bin_size=500):
    """
    - valor: coordenada X o Y del camión
    - bin_size: tamaño del rango para cada bin
    
    min_x = 200287
    max_x = 204707
    bin_size = 500

    min_bin_x = int(min_x // bin_size)  # 400
    max_bin_x = int(max_x // bin_size)  # 409

    num_bins_x = max_bin_x - min_bin_x + 1  # 409 - 400 + 1 = 10

    """
    if valor is None or valor < 0:
        return -1
    return int(valor // bin_size)  # Ej: 200287 // 500 = 400


# Discretiza ETA en grupos aproximados
def discretizar_eta(eta):
    if eta is None:
        return -1
    if eta < 3:
        return 0
    elif eta < 6:
        return 1
    elif eta < 9:
        return 2
    elif eta < 12:
        return 3
    else:
        return 4

#======================================
# Función principal: construir estados
#======================================
def state_builder3(tick_info: dict, ordered_shovel_names=None) -> List[Tuple]:
    """
    Construye una lista de estados para todos los camiones en un tick.

    Parámetros:
    - tick_info: diccionario que contiene 'truck_states' y 'shovel_states'.
    - ordered_shovel_names: lista fija para mantener el orden de las palas.

    Retorna:
    - states: Lista de tuplas de estados, uno por camión.
    - truck_names: Lista de nombres reales de los camiones.
    """

    truck_dict = tick_info.get("truck_states", {})
    shovel_dict = tick_info.get("shovel_states", {})

    # Establece orden de palas si no fue dado
    if ordered_shovel_names is None:
        ordered_shovel_names = sorted(shovel_dict.keys())

    states = []
    truck_names = []  # Para guardar los nombres reales de los camiones

    # ===================================
    # Construcción de etiquetas una sola vez
    # ===================================
    state_labels = [
        "pos_x_discr",
        "pos_y_discr",
        "status_camion",
        "fuel_discr"
    ]

    # Primero todos los ETA
    for shovel_name in ordered_shovel_names:
        state_labels.append(f"eta_{shovel_name}")

    # Luego todos los Valor_tiempo_Nuevo
    for shovel_name in ordered_shovel_names:
        state_labels.append(f"Valor_tiempo_Nuevo_{shovel_name}")
    
    # Variables de pala: state, queue_count, priority, coverage
    pala_attrs = ["state", "queue_count", "priority", "coverage"]
    for attr in pala_attrs:
        for shovel_name in ordered_shovel_names:
            state_labels.append(f"{attr}_{shovel_name}")


    # ===================================
    # Construcción de estados
    # ===================================
    for truck_id, truck_info in truck_dict.items():
        truck_names.append(truck_id)  # Guardar el nombre real del camión
        
        #===============================
        # Parte 1: Variables del camión
        #===============================
        position = truck_info.get("position", [-1, -1])
        x = position[0] if len(position) > 0 else -1
        y = position[1] if len(position) > 1 else -1
        status = truck_info.get("status", "")
        fuel = truck_info.get("tank_fuel_level", None)

        # Construye estado base
        state = [
            discretizar_position(x, bin_size=500),  # Posicion X del camion
            discretizar_position(y, bin_size=500),  # Posicion Y del camion
            status_mapping.get(status, -1),  # Status del Camion
            discretizar_fuel(fuel)   # Nivel de combustible del camion
        ]

        #=================================================
        # Parte 1.2: Variables del camión: info de ETA hacia palas ACTIVAS
        #=================================================
        eta_dict = truck_info.get("ETA", {})
        eta_values = []
        tiempo_nuevo_values = []

        # eta_PH002, eta_EX004, ..., eta_CF002: ETA discretizada hacia cada pala (si está activa, si no, -1)
        for shovel_name in ordered_shovel_names:
            shovel_info = shovel_dict.get(shovel_name, {})
            shovel_state = shovel_info.get("state", -1)

            if shovel_state == 1:  # Solo considerar palas activas
                # ETA desde el camión a esta pala
                eta_value = eta_dict.get(shovel_name, None)
                eta_discreta = discretizar_eta(eta_value)

                # shovel_spot_time y shovel_cycle desde shovel_states
                shovel_spot_time = shovel_info.get("shovel_spot_time", 0) or 0
                shovel_cycle = shovel_info.get("shovel_cycle", 0) or 0

                # Nueva variable: ETA - (shovel_spot_time + shovel_cycle)
                if eta_value is not None:
                    Variable_Tiempo_Nuevo = eta_value - (shovel_spot_time + shovel_cycle)
                else:
                    Variable_Tiempo_Nuevo = -1

            else:
                eta_discreta = -1  # No incluir ETA si la pala no está activa
                Variable_Tiempo_Nuevo = -1
            
            # Agregar ETA y la nueva variable al estado
            #state.append(eta_discreta)
            #state.append(Variable_Tiempo_Nuevo)

            eta_values.append(eta_discreta)
            tiempo_nuevo_values.append(Variable_Tiempo_Nuevo)

            #if not state_labels or len(state_labels) < len(state):
            #    state_labels.extend([
            #        f"eta_{shovel_name}",
            #        f"Valor_tiempo_Nuevo_{shovel_name}"
            #    ])

        # Añadir todos los ETA juntos y luego todos los tiempos nuevos
        state.extend(eta_values)
        state.extend(tiempo_nuevo_values)

        #===============================
        # Parte 2: Variables del Accion(Pala)
        #===============================
        # Agrega variables de palas en orden, agregadas al estado base
        #for attr in ["priority", "coverage", "main_state","state","queue_count"]:
        for attr in ["state","queue_count", "priority", "coverage"]: 
            for shovel_name in ordered_shovel_names:
                #shovel_info = shovel_dict.get(shovel_name, {})
                #value = shovel_info.get(attr, -1)
                value = shovel_dict.get(shovel_name, {}).get(attr, -1)
                state.append(value)
        # shovel_state_PH002, ..., shovel_state_CF002 : Estado de cada pala (0 = inactiva, 1 = activa, etc.)
        # queue_PH002, ..., queue_CF002	Tamaño de cola de cada pala
        states.append(tuple(state))
    
    return states, truck_names, state_labels

"""
#########################################################################################################################################################################
def validate_state_builder(json_path, tick_id="3"):
    with open(json_path, "r") as f:
        tick_json = json.load(f)

    if tick_id not in tick_json:
        raise ValueError(f"Tick {tick_id} no encontrado en el JSON.")

    tick_data = tick_json[tick_id]

    if "shovel_states" not in tick_data:
        raise KeyError(f"'shovel_states' no está presente en el tick {tick_id}.")

    available_shovels = list(tick_data["shovel_states"].keys())
    print(f"Palas disponibles en el tick {tick_id}: {available_shovels}")

    states, truck_names, state_labels = state_builder3(tick_data, available_shovels)

    print("\nEtiquetas de los estados:")
    print(f"({', '.join(state_labels)})\n")

    for s, truck_name in zip(states, truck_names):
        print(f"Estado del camión {truck_name}: {s}")

    return state_labels, states, truck_names


# Ejecución
#TICK_JSON_REAL_PATH = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\MINE-hudbay-ALGO-TIME_8_8_2025.json"
TICK_JSON_REAL_PATH = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\MINE-hudbay-ALGO-2025-08-15.json"
validate_state_builder(TICK_JSON_REAL_PATH, tick_id="3")
"""