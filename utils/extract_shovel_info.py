# -------------------------------
# ========== Preparar información de las palas ==========
# -------------------------------
def extract_info_shovel_from_config(config_data_base):
    # Crear un diccionario vacío para almacenar información de todas las palas encontradas en el archivo de configuración
    shovels_info = {}
    # Recorremos cada sitio de carga (load site) definido en el archivo mine_config.json
    for loadsite_id, loadsite_data in config_data_base["load_sites"].items():

        # Accedemos a la sección de "shovels" (palas) dentro del sitio de carga
        shovels = loadsite_data.get("shovels", {}) 
        
        # Recorremos los sitios de carga (load_sites) del JSON, y dentro de cada uno, buscamos las palas.
        for shovel_id, shovel_data in shovels.items():
            # Extraemos el nombre de la pala (ej. "PH001")
            shovel_name = shovel_data.get("name")
            if shovel_name:  # solo si el nombre existe
                # Se construye un diccionario shovels_info para cada pala, guardando su prioridad, cobertura, si está habilitada y su estado principal
                shovels_info[shovel_name] = {
                    "priority": shovel_data.get("priority", -1),
                    "coverage": shovel_data.get("coverage", -1),
                    "enabled": shovel_data.get("enabled", False),
                    "main_state": shovel_data.get("main_state", -1)
                }
    return shovels_info