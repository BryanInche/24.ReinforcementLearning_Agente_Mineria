# Validación del Agente RL en Producción

Este directorio contiene scripts para validar el comportamiento del agente de Reinforcement Learning en un entorno de producción con datos reales.

## 📁 Archivos Disponibles

### 1. `Ejecucion_Agente_production.py` (Script Principal)
**Propósito**: Validación completa del agente RL en producción con datos reales.

**Características**:
- ✅ Usa datos reales de la mina (`MINE-hudbay-TicksTrainRL-TIME-2025-08-01.json`)
- ✅ Carga la Q-table entrenada (`q_table_real.pkl`)
- ✅ Detecta y registra estados nuevos vs conocidos
- ✅ Valida la actualización de la matriz Q
- ✅ Proporciona métricas detalladas de rendimiento
- ✅ Logging completo de todas las decisiones
- ✅ **NUEVO**: Análisis detallado de ETAs y rewards como justificación
- ✅ **NUEVO**: Comparación de todas las opciones disponibles
- ✅ **NUEVO**: Ranking de decisiones basado en rewards esperados

**Uso**:
```bash
python test/Ejecucion_Agente_production.py
```

**Salida**:
- Métricas detalladas de rendimiento
- Análisis de estados nuevos vs conocidos
- Verificación de actualización de Q-table
- Logs completos en `logs/agente_rl_produccion_YYYYMMDD_HHMMSS.log`

### 2. `validacion_estados_nuevos.py` (Análisis Específico)
**Propósito**: Análisis detallado del comportamiento con estados nuevos vs conocidos.

**Características**:
- 🔍 Compara rendimiento entre estados nuevos y conocidos
- 📊 Estadísticas de exploración vs explotación
- 💰 Análisis de recompensas por tipo de estado
- 📝 Ejemplos de estados nuevos encontrados

**Uso**:
```bash
python test/validacion_estados_nuevos.py
```

### 3. `test_agente_produccion.py` (Prueba Rápida)
**Propósito**: Prueba rápida y simple del agente en producción.

**Características**:
- ⚡ Procesa solo 10 ticks para prueba rápida
- 🧪 Salida detallada paso a paso
- 📈 Métricas básicas de rendimiento
- 🔍 Verificación de actualización de Q-table

**Uso**:
```bash
python test/test_agente_produccion.py
```

### 4. `prueba_agente_con_etas.py` (Análisis Detallado de ETAs)
**Propósito**: Análisis detallado de decisiones con información de ETAs y rewards.

**Características**:
- 🚗 **NUEVO**: Muestra ETAs del camión hacia cada pala
- 💰 **NUEVO**: Calcula rewards esperados para todas las opciones
- 📊 **NUEVO**: Ranking de decisiones basado en rewards
- ⚖️ **NUEVO**: Análisis ETA vs Reward (si seleccionó la mejor opción)
- 🔍 **NUEVO**: Comparación detallada de todas las palas disponibles
- 📋 Tabla comparativa con Q-values, ETAs, estado de palas y rewards

**Uso**:
```bash
python test/prueba_agente_con_etas.py
```

## 🎯 Objetivos de Validación

### 1. Validar Estados Nuevos
- **Objetivo**: Verificar que el agente puede manejar estados que no vio durante el entrenamiento
- **Métrica**: Número de estados nuevos encontrados y agregados a la Q-table
- **Resultado esperado**: El agente debe poder procesar estados nuevos y agregarlos a su conocimiento

### 2. Validar Análisis de ETAs y Rewards
- **Objetivo**: Verificar que el agente toma decisiones considerando ETAs y rewards esperados
- **Métrica**: Comparación entre mejor opción por reward vs opción seleccionada
- **Resultado esperado**: El agente debe seleccionar opciones con buenos ETAs y rewards

### 3. Validar Actualización de Q-Table
- **Objetivo**: Confirmar que la matriz Q se actualiza correctamente con nueva información
- **Métrica**: Número de Q-values actualizados durante la ejecución
- **Resultado esperado**: Los Q-values deben cambiar basándose en las recompensas obtenidas

### 4. Validar Comportamiento Exploración/Explotación
- **Objetivo**: Verificar el balance entre exploración y explotación
- **Métrica**: Tasa de exploración vs explotación
- **Resultado esperado**: El agente debe explorar estados nuevos y explotar estados conocidos

### 5. Validar Rendimiento en Producción
- **Objetivo**: Medir el rendimiento del agente con datos reales
- **Métrica**: Recompensa total y promedio por decisión
- **Resultado esperado**: El agente debe mantener o mejorar su rendimiento

## 📊 Métricas Clave

### Métricas Generales
- **Total de ticks procesados**: Número de ticks analizados
- **Total de camiones optimizados**: Camiones que requirieron decisión del agente
- **Total de camiones fijos**: Camiones que no necesitaron optimización
- **Estados únicos vistos**: Diferentes estados encontrados

### Métricas de Aprendizaje
- **Estados nuevos encontrados**: Estados no vistos durante el entrenamiento
- **Q-values actualizados**: Número de actualizaciones en la matriz Q
- **Decisiones por exploración**: Acciones tomadas aleatoriamente
- **Decisiones por explotación**: Acciones basadas en Q-values conocidos

### Métricas de Rendimiento
- **Recompensa total acumulada**: Suma de todas las recompensas obtenidas
- **Recompensa promedio por camión**: Rendimiento promedio por decisión
- **Distribución de acciones por pala**: Cómo se distribuyen las asignaciones

## 🔧 Configuración

### Rutas de Archivos
```python
# Q-table entrenada
QTABLE_PATH = "agent/q_table_real.pkl"

# Datos reales de producción
TICK_FILE = "agent/MINE-hudbay-TicksTrainRL-TIME-2025-08-01.json"

# Palas disponibles
SHOVEL_NAMES = ['PH002', 'EX004', 'PH003', 'PH001', 'CF001', 'CF002']
```

### Parámetros del Agente
- **Alpha (tasa de aprendizaje)**: 0.1
- **Gamma (factor de descuento)**: 0.9
- **Epsilon (tasa de exploración)**: 1.0
- **Epsilon decay**: 0.1
- **Epsilon min**: 0.01

## 📝 Interpretación de Resultados

### ✅ Comportamiento Esperado
1. **Estados nuevos**: El agente debe encontrar y procesar estados nuevos
2. **Actualización Q-table**: Los Q-values deben actualizarse con nueva información
3. **Exploración balanceada**: Mezcla de exploración y explotación
4. **Rendimiento estable**: Recompensas consistentes o mejorando

### ⚠️ Señales de Problemas
1. **No se encuentran estados nuevos**: Posible sobreajuste o datos muy similares
2. **Q-values no se actualizan**: Problema en el algoritmo de actualización
3. **Solo exploración**: Epsilon demasiado alto
4. **Solo explotación**: Epsilon demasiado bajo
5. **Recompensas muy bajas**: Problema en la función de recompensa

## 🚀 Próximos Pasos

1. **Ejecutar validación completa**: Usar `Ejecucion_Agente_production.py`
2. **Analizar resultados**: Revisar métricas y logs
3. **Ajustar parámetros**: Si es necesario, modificar epsilon, alpha, etc.
4. **Validar con más datos**: Probar con diferentes conjuntos de datos
5. **Monitoreo continuo**: Implementar validación automática en producción

## 📞 Soporte

Para problemas o preguntas sobre la validación:
1. Revisar los logs generados
2. Verificar que los archivos de datos existen
3. Confirmar que la Q-table está correctamente entrenada
4. Revisar las métricas de rendimiento 