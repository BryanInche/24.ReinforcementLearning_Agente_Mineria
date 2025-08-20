import numpy as np
import random
# =========================
# Clase del agente Q-Learning
# =========================

r"""
Q(st,at)←Q(st,at)+α⋅[r+γ⋅amaxQ(st+1,a)−Q(st,at)]
Donde:
•	sts_tst: estado actual
•	ata_tat: acción tomada
•	rrr: recompensa obtenida
•	st+1s_{t+1}st+1: nuevo estado alcanzado
•	γ\gammaγ: factor de descuento
•	α\alphaα: tasa de aprendizaje
"""

class QLearningAgent:
    #def __init__(self, actions, state_size, alpha=0.1, gamma=0.9,epsilon=1.0, epsilon_decay=0.1, epsilon_min=0.01):
    def __init__(self, actions, alpha=0.1, gamma=0.9,epsilon=0.7, epsilon_decay=0.1, epsilon_min=0.01):
        # Lista de acciones que el agente puede elegir.
        self.actions = actions  # acciones posibles como nombres de palas, ej: ["PH001", "PH002"]
        
        # Referencia al número de características (features) que definen un estado.
        # Sirve como referencia informativa, por ejemplo, si planeas exportar el modelo o saber cuántas features usas para representar un estado.
        #self.state_size = state_size  # tamaño del estado (no lo usamos directamente, pero se mantiene como referencia)
        
        # Tabla de valores Q
        self.q_table = dict()  # clave: estado (tuple), valor: dict con Q para cada acción (nombre de pala)
        """
        (68108, 2, 1): {"PH001": 0.3, "PH002": -1.1},
        (68075, 0, 0): {"PH001": 1.2, "PH002": 0.5},
        ...
        }
        """
        
        # cuánto se ajusta el valor Q con cada nueva experiencia: cercano a 1 actualiza muy rápido; uno cercano a 0 actualiza muy lento
        self.alpha = alpha  # tasa de aprendizaje
        # gamma = 0 → el agente solo valora la recompensa inmediata.
        # gamma = 1 → el agente valora recompensas futuras tanto como las inmediatas.
        
        self.gamma = gamma  # factor de descuento
        
        """
        # trade-off entre exploración y explotación:
        Si epsilon = 1, el agente elige todo aleatorio.(solo exploracion)
        Si epsilon = 0, el agente solo elige la mejor acción actual (según la Q-table).
        """
        
        ## epsilon = 1.0 -> 100% exploración, A medida que epsilon baja, explora menos y explota más lo aprendido.
        self.epsilon = epsilon  # tasa de exploración
        
        #Esto reduce poco a poco la exploración, define qué tan rápido baja epsilon. Cuanto más pequeño, más lento decae.
        self.epsilon_decay = epsilon_decay

        # Límite inferior de exploración.
        # Evita que el agente deje de explorar completamente.
        self.epsilon_min = epsilon_min


    """
    1. Convierte un estado representado como diccionario (dict) en una tupla,
    para poder usarlo como clave única en la Q-table.

    Ejemplo:
    Input:
        state_dict = {
            "locacion_camion": 68108,
            "cola_en_pala_PH001": 2,
            "cola_en_pala_PH002": 1
        }
    Salida:
        (68108, 2, 1)

    Este formato es esencial para que el agente aprenda por estado específico.
    """
    #def get_state_key(self, state_dict):
    #    return tuple(state_dict.values())


    """
    2. Selecciona la mejor acción posible para un estado dado usando una política ε-greedy.

    Parámetros:
        state: dict con la información del estado actual del camión.
        valid_actions: lista de nombres de palas habilitadas en ese instante.

    Salida:
        Acción seleccionada (ejemplo: "PH001").
    """
    def choose_action(self, state, valid_actions):
        #key = self.get_state_key(state)  # Era necesario cuando state era diccionario, y necesitabamos una tupla
        key = state  # state ya es una tupla

        #Si el estado no se ha sido visto nunca antes, lo agrega a la tabla Q, pero le asigna Q-valores 0.0 para todas las acciones válidas
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in valid_actions}

        # Exploración (Nos permite que el agente explore todo el entorno sin importar si obtiene recompensas)
        if random.random() < self.epsilon:
            self.last_action_was_random = True
            return random.choice(valid_actions)

        # Explotación (El agente solo realiza las acciones que le generan altas recompensas)
        self.last_action_was_random = False
        valid_qs = {a: self.q_table[key].get(a, 0.0) for a in valid_actions}
        return max(valid_qs, key=valid_qs.get)


    r"""
    3. Actualiza la Q-table según la fórmula de Q-learning:
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

    Parámetros:
        state: dict del estado actual (antes de tomar la acción)
        action: acción tomada (ejemplo: "PH001")
        reward: recompensa obtenida (ej: tiempo eficiente, penalidad por espera, etc.)
        next_state: dict del estado siguiente después de ejecutar la acción
        next_valid_actions: acciones posibles en el nuevo estado

    Lógica:
    - Si el estado actual o el siguiente no están en la tabla Q, los inicializa.
    - Calcula el valor Q actual (`q_predict`) y el valor objetivo (`q_target`).
    - Aplica la fórmula para actualizar el Q-valor.
        """
    def update(self, state, action, reward, next_state, next_valid_actions):
        #state_key = self.get_state_key(state)
        state_key = state  #state ya es una tupla

        #next_key = self.get_state_key(next_state)
        next_key = next_state # state ya es una tupla

        # Paso 1: Inicializar el estado actual si no existe en la Q-table
        # Si el estado no existe, se inicializa con todas las acciones posibles (self.actions) y valor Q = 0.0
        if state_key not in self.q_table:
            print(f"[Estado] Agregado: {state_key}")
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        
        # Paso 2: Inicializar el siguiente estado si no existe en la Q-table
        """
        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0.0 for a in next_valid_actions}
        """

        if next_key not in self.q_table:
            if next_valid_actions:
                print(f"[Siguiente Estado] Agregado: {next_key}")
                self.q_table[next_key] = {a: 0.0 for a in next_valid_actions}
            else:
                # Asegura que el estado al menos exista (sin acciones válidas)
                print(f"[Siguiente Estado sin acciones válidas: {next_key}")
                self.q_table[next_key] = {}

        
        # Paso 3: Obtener el valor Q actual (Q(s,a))
        # Q(s,a) ← Q(s,a) + α [r + γ * maxₐ' Q(s', a') − Q(s,a)]
        q_predict = self.q_table[state_key].get(action, 0.0)  # Valor actual Q(s,a) antes de la actualización
        
        # Paso 4: Calcular el valor objetivo (Q-target)
        q_target = reward   # Recompensa inmediata r
        # Mejor valor Q para el siguiente estado s' y cualquier acción a' (maxₐ' Q(s', a'))
        # self.gamma : Factor de descuento
        # self.alpha : Tasa de aprendizaje
        # Se evalúan todas las acciones en next_valid_actions
        if next_valid_actions:
            q_target += self.gamma * max(  #max(...): encuentra la mejor acción futura esperada, es decir, max Q(s', a')
                # next_valid_actions: lista de acciones posibles en el siguiente estado siguiente(s')
                [self.q_table[next_key].get(a, 0.0) for a in next_valid_actions]
            ) #¡solo actualizas el Q de la acción que tomaste en el estado actual! Las demás no se tocan.

        # Actualización del valor Q(s,a) en la tabla
        self.q_table[state_key][action] = q_predict + self.alpha * (q_target - q_predict)

    
    """
    4. Reduce progresivamente la tasa de exploración (epsilon), 
    para que el agente dependa más de su conocimiento aprendido
    a medida que pasa el tiempo (menos aleatoriedad).

    Esto asegura que:
    - Al inicio explore mucho.
    - A medida que aprende, explora menos y explota(tomar acciones que solo obtienen recompsensas altas) más.
    """
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay