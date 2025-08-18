import random 
def generar_tick_json_artificial(shovel_names, num_ticks=100, num_trucks=5):
    tick_json = {}

    for tick in range(num_ticks):
        truck_states = {}
        for i in range(num_trucks):
            truck_id = f"CM{str(i+1).zfill(2)}"
            truck_states[truck_id] = {
                "name": truck_id,
                "time": tick,
                "state": "READY",
                "status": random.choice([
                    "waiting for shovel", "loading", "unloading",
                    "moving load", "moving unload", "waiting for dumper"
                ]),
                "position": [random.uniform(200000, 210000), random.uniform(8390000, 8400000)],
                "tank_fuel_level": random.randint(10, 100),
                "type": random.choice(["interp", "interp2", "NODE", "interp1"])
            }

        shovel_states = {}
        for name in shovel_names:
            shovel_states[name] = {
                "name": name,
                "time": tick,
                "state": random.choice([1, 3, 4]),  # 1: activa
                "position": [
                    round(random.uniform(200000, 210000), 2),
                    round(random.uniform(8390000, 8400000), 2)
                ],
                "queue_count": random.choice([0, 1, 2])
            }

        tick_json[str(tick)] = {
            "tick": tick,
            "truck_states": truck_states,
            "shovel_states": shovel_states
        }

    return tick_json