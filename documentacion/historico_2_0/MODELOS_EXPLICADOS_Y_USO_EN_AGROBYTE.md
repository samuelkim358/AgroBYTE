# Guía docente — Algoritmos de AgroBYTE y utilización en el reto

> Para estudiantes de Ciencias de la Computación y Ciencia de Datos. No se requiere experiencia previa en Machine Learning.

## Parte I. Qué son los modelos y cómo funcionan en general

### Conceptos que compartimos

- **Entrada (`X`)**: columnas que describen cada parcela, como latitud, NDVI, temperatura o precipitación.
- **Objetivo (`y`)**: rendimiento observado en toneladas de cebada por hectárea (`RENDIMIENTO_T_HA`).
- **Entrenamiento**: el algoritmo aprende relaciones entre `X` e `y` de las 138 parcelas etiquetadas.
- **Inferencia**: aplica esas relaciones a parcelas cuyo rendimiento todavía desconocemos; no puede calcular ahí el error real.
- **Generalización**: que las reglas aprendidas funcionen también en parcelas que no participaron en el entrenamiento.

### 1. Regresión Ridge — referencia lineal regularizada

Intenta representar el rendimiento como combinación ponderada de características:

`ŷ = β0 + β1·x1 + β2·x2 + ... + βp·xp`

Minimiza errores cuadrados más una penalización a coeficientes grandes:

`minβ  Σ(yi − ŷi)² + α Σ βj²`

**Por qué nos interesa:** es un punto de comparación simple, permite identificar si los modelos de árboles realmente aportan mejora. Necesita normalización de las variables y codificación de categorías.

### 2. Extra Trees — muchos árboles con aleatoriedad

Un árbol divide los datos en grupos según sus variables (por ejemplo, región, latitud o NDVI). Extra Trees entrena muchos árboles introduciendo aleatoriedad en las divisiones y promedia los resultados:

`ŷ(x) = [T1(x) + T2(x) + ... + TM(x)] / M`

**Por qué nos interesa:** aprende relaciones no lineales con poco preprocesamiento. **Riesgo:** puede depender demasiado de zonas parecidas a las de entrenamiento y funcionar peor en nuevos municipios.

### 3. CatBoost — boosting de árboles

Crea árboles secuencialmente. Cada nuevo árbol intenta corregir lo que la combinación anterior todavía no explica:

`Fm(x) = Fm−1(x) + η·hm(x)`

Aquí `η` es la tasa de aprendizaje y `hm` el nuevo árbol. Su implementación usa regularización y técnicas adicionales para tratar datos tabulares.

**Por qué nos interesa:** puede capturar relaciones complejas entre clima, estado e índices. **Riesgo:** con apenas 138 etiquetas, demasiados árboles o profundidad alta pueden sobreajustar.

### 4. HistGradientBoosting — boosting eficiente por intervalos

En lugar de buscar continuamente todos los cortes numéricos, agrupa características en intervalos o histogramas y construye árboles secuenciales. Permite regularizar profundidad, tamaño de hojas y número de iteraciones.

**Por qué nos interesa:** otro punto de comparación no lineal; no asumimos que superará a los demás.

### 5. Media — el modelo más simple

Predice siempre el promedio de los rendimientos disponibles **en su pliegue de entrenamiento**. Es importante como referencia: un algoritmo avanzado debe mejorar esta estrategia trivial.

## Parte II. Cómo usamos esos modelos específicamente en AgroBYTE

### Preparación de los datos

1. Los archivos Básico y PRO incluyen muchas fechas por parcela; filtramos por temporada 2025 y conservamos los sensores separados.
2. Resumimos índices por parcela, mes y satélite para construir una matriz tabular.
3. Extraemos precipitación, temperatura y topografía de archivos ráster georreferenciados.
4. Formamos una fila por parcela. Separamos las 138 etiquetas visibles de las 59 predicciones sin rendimiento.
5. El filtrado de nubosidad del 20% es una decisión **exploratoria**, no un umbral universal demostrado.

### Grupos de características ensayados

| Grupo | Qué incluimos | Hipótesis |
|---|---|---|
| `geografia` | Área, latitud, longitud, estado, municipio | La ubicación explica diferencias regionales |
| `completo` | Geografía, satélites, clima y topografía | Más contexto puede mejorar la predicción |
| `reducido` | Geografía, clima, topografía e índices seleccionados | Un subconjunto puede reducir ruido/redundancia |

### Configuraciones reales de la Fase 2

| Candidato | Modelo | Variables |
|---|---|---|
| `geo_extratrees` | Extra Trees | Geografía |
| `completo_extratrees` | Extra Trees | Completo |
| `reducido_extratrees` | Extra Trees | Reducido |
| `completo_catboost` | CatBoost | Completo |
| `reducido_catboost` | CatBoost | Reducido |
| `reducido_histboost` | HistGradientBoosting | Reducido |
| `reducido_ridge` | Ridge | Reducido |

### Preprocesamiento correcto dentro de la validación

Se usa `Pipeline`/`ColumnTransformer` para que la **mediana de imputación**, las **categorías observadas** y la **escala** (en Ridge) se aprendan con las parcelas del entrenamiento de cada pliegue. No calculamos esas estadísticas con rendimientos de evaluación ocultos.

### ¿Cómo escogemos el ganador?

- **Prueba A: validación por parcelas.** Repetimos cinco particiones con semillas 42, 73 y 2026. Medimos el error en parcelas reservadas; las mismas parcelas pueden aparecer en distintos ensayos, pero nunca entrenamos un pliegue con su propio objetivo de prueba.
- **Prueba B: municipios enteros.** `GroupKFold` reserva zonas completas; evalúa extrapolación geográfica, más difícil que predecir parcelas próximas.
- **Prueba C: validación anidada.** Seleccionamos una familia en particiones internas y evaluamos sobre particiones externas no vistas en esa selección. La exploración previa sobre el mismo dataset limita cuán independiente puede considerarse esta comprobación.
- **Selección final pendiente.** No entregamos todavía como definitivas las 59 predicciones experimentales.

### Resultados que podemos citar con cuidado

| Estudio | RMSE | MAE | R² |
|---|---:|---:|---:|
| Fase 1 — Extra Trees geografía, validación por parcela | 0.496 | 0.356 | 0.657 |
| Fase 2 — selección anidada, validación por parcela | 0.503 | 0.351 | 0.648 |

**No son las métricas oficiales de FIRA:** el concurso tiene 59 rendimientos reales ocultos. Además, estos resultados se obtuvieron explorando las 138 etiquetas disponibles, por lo que no deben confundirse con la precisión futura en cualquier municipio de México.

### Dónde encontrar cada parte en el código

- `codigo/experimento_agrocebada.py`: lectura, filtrado, extracción, ingeniería de características y modelos iniciales.
- `codigo/fase2_modelos_comparacion.py`: configuración de candidatos, entrenamiento por fold, selección y modelo experimental.
- `codigo/diagnosticos_fase2.py`: pruebas de integridad de las 59 salidas y diagnóstico de errores por estado.
- `salidas/fase1` y `salidas/fase2`: métricas y matrices que se vuelven a generar al ejecutar el programa.

### Preguntas para que el equipo compruebe lo aprendido

1. ¿Por qué 155,470 observaciones satelitales no equivalen a 155,470 rendimientos independientes?
2. ¿Qué sucede si las mediciones de la misma parcela se dividen accidentalmente entre entrenamiento y validación?
3. ¿Por qué Extra Trees puede ganar en parcelas próximas y CatBoost en municipios no vistos?
4. ¿Qué significan `RMSE=0.503 t/ha` y `R²=0.648`? ¿Por qué `R²` no es porcentaje de aciertos?
5. ¿Por qué el modelo geográfico no basta para predecir parcelas en lugares sin ejemplos similares?
