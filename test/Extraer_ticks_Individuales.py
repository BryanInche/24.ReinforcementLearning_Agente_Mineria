import json
from pathlib import Path
import os 

# Ruta del archivo con todos los ticks
TICK_FILE = Path(r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\agent\MINE-hudbay-TicksTrainRL-TIME-2025-08-01.json")

# Ruta donde guardaremos los ticks filtrados
OUTPUT_DIR = r"C:\Simluador_Opt_GRUPAL\Simulador_Inteligente\MVP1\src_new\algorithms\RL_model\ticks_filtrados"

# 1️. Leer archivo grande
with open(TICK_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ver el tipo y llaves principales
print("Tipo raíz:", type(data))
if isinstance(data, dict):
    print("Llaves:", list(data.keys())[:10])
elif isinstance(data, list):
    print("Cantidad de elementos:", len(data))

# 2️. Filtrar un solo tick (ejemplo: el tick número 5)
# 0, 3, 6, 9, 12 --- NO EXISTEN CAMIONES PARA OPTIMIZAR
# 195, 201, 213, 216, 222  -- EXISTEN CAMIONES PARA OPTIMIZAR
tick_index = "222"  # Puedes cambiarlo
tick_data = data[tick_index]

# 3️. Guardar el tick en un JSON separado
output_path = os.path.join(OUTPUT_DIR, f"tick_{tick_index}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tick_data, f, ensure_ascii=False, indent=4)

print(f"Tick {tick_index} guardado en {output_path}")
