# rewards.py
"""
Este módulo define la función de recompensa que utiliza el agente RL.

Puedes modificar esta función para ajustar las penalizaciones o premios según tu dominio.

"""
def calcular_recompensa(status, action, shovels_info, truck_etas=None):
    """
    Calcula la recompensa para una acción del agente RL.
    
    Parámetros:
    - status: estado actual del camión
    - action: pala seleccionada
    - shovels_info: información de todas las palas
    - truck_etas: diccionario con ETAs del camión hacia cada pala {shovel_name: eta_value}
    """
    reward = 0

    shovel = shovels_info.get(action)

    if shovel is None:
        return -20  # acción completamente inválida

    state = shovel.get("state", -1)
    queue_count = shovel.get("queue_count", 0)

    # 1. Penaliza elegir una pala que no está activa
    if state != 1:     
        reward -= 10   

    # 2. Penalización por colas largas
    if queue_count >= 3:
        reward -= 4
    elif queue_count == 2:
        reward -= 2
    elif queue_count == 1:
        reward += 1  # manejable
    else:
        reward += 3  # excelente: sin cola

    # 3. Recompensa si el camión está esperando y elegimos una pala activa
    if status == "waiting for shovel" and state == 1:
        reward += 8

    # 4. Penaliza mover el camión innecesariamente
    if status in ["loading", "moving load", "moving unload"]:
        reward -= 3

    # 5. Bonus por reducir "hang time": pala activa y sin cola
    if state == 1 and queue_count == 0 and status == "waiting for shovel":
        reward += 5

    # 6. NUEVA LÓGICA: Recompensas basadas en ETA
    if truck_etas is not None and action in truck_etas:
        eta_to_shovel = truck_etas[action]
        
        if eta_to_shovel is not None and eta_to_shovel >= 0:
            # Recompensa por ETA eficiente (menor tiempo = mayor recompensa)
            if eta_to_shovel < 3:
                reward += 8  # Muy cerca, excelente elección
            elif eta_to_shovel < 6:
                reward += 5  # Distancia media, buena elección
            elif eta_to_shovel < 9:
                reward += 2  # Distancia aceptable
            elif eta_to_shovel < 12:
                reward -= 1  # Distancia larga, penalización leve
            else:
                reward -= 3  # Muy lejos, penalización mayor
            
            # Bonus adicional: si es la pala con menor ETA disponible
            if state == 1:  # Solo considerar palas activas
                available_etas = {shovel_name: eta for shovel_name, eta in truck_etas.items() 
                                if shovels_info.get(shovel_name, {}).get("state") == 1 
                                and eta is not None and eta >= 0}
                
                if available_etas:
                    min_eta = min(available_etas.values())
                    if eta_to_shovel == min_eta:
                        reward += 4  # Bonus por elegir la pala más cercana
                    elif eta_to_shovel <= min_eta * 1.2:  # Dentro del 20% del mejor ETA
                        reward += 2  # Bonus por estar cerca del óptimo

    return reward

"""

#=================================================
# Función de prueba para demostrar la lógica de ETA
#=================================================
def probar_recompensas_con_eta():
    #Función de prueba para demostrar cómo funciona la nueva lógica de recompensas con ETA
    print("=== PRUEBA DE RECOMPENSAS CON ETA ===\n")
    
    # Simular información de palas
    shovels_info = {
        "PH002": {"state": 1, "queue_count": 0},  # Activa, sin cola
        "PH003": {"state": 1, "queue_count": 2},  # Activa, cola de 2
        "PH001": {"state": 1, "queue_count": 1},  # Activa, cola de 1
        "CF002": {"state": 4, "queue_count": 0},  # Inactiva
        "EX004": {"state": 4, "queue_count": 0},  # Inactiva
        "CF001": {"state": 4, "queue_count": 0}   # Inactiva
    }
    
    # Simular ETAs de un camión hacia cada pala
    truck_etas = {
        "PH002": 2.5,   # Muy cerca
        "PH003": 8.7,   # Distancia media
        "PH001": 9.4,   # Distancia media-alta
        "CF002": 12.7,  # Lejos
        "EX004": None,  # Sin ETA
        "CF001": None   # Sin ETA
    }
    
    # Casos de prueba
    casos_prueba = [
        ("waiting for shovel", "PH002", "Caso 1: Pala más cercana (ETA=2.5)"),
        ("waiting for shovel", "PH003", "Caso 2: Pala media (ETA=8.7)"),
        ("waiting for shovel", "PH001", "Caso 3: Pala lejana (ETA=9.4)"),
        ("waiting for shovel", "CF002", "Caso 4: Pala inactiva (ETA=12.7)"),
        ("moving unload", "PH002", "Caso 5: Camión en movimiento hacia pala cercana"),
        ("loading", "PH002", "Caso 6: Camión cargando (no debería moverse)")
    ]
    
    for status, action, descripcion in casos_prueba:
        # Sin ETA (función original)
        reward_sin_eta = calcular_recompensa(status, action, shovels_info)
        
        # Con ETA (función nueva)
        reward_con_eta = calcular_recompensa(status, action, shovels_info, truck_etas)
        
        print(f"{descripcion}")
        print(f"  Recompensa sin ETA: {reward_sin_eta}")
        print(f"  Recompensa con ETA: {reward_con_eta}")
        print(f"  Diferencia: {reward_con_eta - reward_sin_eta}")
        print()


if __name__ == "__main__":
    probar_recompensas_con_eta()
"""
