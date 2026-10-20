# Fase 3: por qué probamos XGBoost, cómo lo evaluamos y qué aprendimos

## A. ¿Qué estamos prediciendo?

Una parcela tiene características `x` (estado, municipio, NDVI, clima, altitud, etc.). Con las **138 etiquetas** aprendemos una función de predicción del rendimiento en **toneladas por hectárea**: `ŷ = f(x)`.

Al analizar cientos de observaciones por parcela, nuestras matrices finales tienen **197 filas**: una por parcela. La variable objetivo solo está presente en 138 filas. Es un error tratar miles de observaciones satelitales repetidas de una misma parcela como si fueran miles de rendimientos independientes.

## B. Los tres algoritmos

**Extra Trees.** Crea muchos árboles de decisión con divisiones parcialmente aleatorizadas. La predicción final es el promedio de sus árboles: `ŷ(x) = (1/T) Σ T_j(x)`. Puede capturar relaciones no lineales y funciona como referencia geográfica.

**CatBoost.** Entrena una secuencia de árboles pequeños que mejora una predicción acumulada. Su representación simplificada es `F_m(x) = F_(m-1)(x) + η h_m(x)`. Comparado con Extra Trees, este procedimiento construye árboles dependientes de las correcciones anteriores.

**XGBoost.** También agrega árboles en secuencia, optimizando una pérdida más regularización. Conceptualmente busca disminuir:

`Objetivo = Σ L(y_i, ŷ_i) + Σ Ω(árbol_m)`

La regularización puede penalizar la complejidad de los árboles. En esta fase comparamos variantes **deliberadamente conservadoras**: profundidad 2, aprendizaje lento, pesos mínimos por hoja y penalizaciones L1/L2. Con 138 parcelas, un modelo muy flexible podría memorizar ruido.

## C. Las métricas

- **MAE:** `(1/n) Σ |y_i - ŷ_i|`; toneladas por hectárea equivocadas en promedio.
- **RMSE:** `√[(1/n) Σ (y_i - ŷ_i)²]`; castiga relativamente más los errores grandes.
- **R²:** `1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²`; compara con predecir siempre la media del conjunto evaluado. Puede ser negativo.

Ninguna métrica puede calcularse de forma honesta para las 59 parcelas ocultas sin conocer sus rendimientos reales.

## D. ¿Qué significa cada validación?

1. **Por parcelas (5 folds × 3 semillas):** en cada fold entrenamos sobre cuatro quintos de las parcelas conocidas y estimamos el quinto restante. Repetimos con semillas 42, 73 y 2026 para observar sensibilidad. Todos los algoritmos usan las **mismas particiones**.
2. **Por municipios (GroupKFold):** dejamos municipios enteros fuera del entrenamiento, una prueba más estricta cuando aplicamos el modelo lejos de terrenos conocidos.
3. **Anidada (5 externos × 3 internos):** ocultamos un fold externo; dentro del entrenamiento externo comparamos cuatro algoritmos y elegimos el de menor RMSE interno. Después lo evaluamos **una sola vez** sobre el fold externo. Reduce optimismo de la búsqueda, pero no crea un test independiente respecto de toda nuestra investigación previa.

**Prevención de fugas de información:** cada `Pipeline` ajusta medianas, transformaciones categóricas y árboles solo con los datos de entrenamiento de cada fold.

## E. Resultados realmente medidos con los insumos de FIRA (15 de octubre de 2026)

| Candidato | RMSE por parcelas: media de 3 semillas | RMSE por municipios |
|---|---:|---:|
| Extra Trees geográfico | **0.49634** | 0.92048 |
| CatBoost reducido | 0.54020 | 0.77251 |
| XGBoost reducido conservador | 0.53374 | **0.67471** |
| XGBoost completo conservador | 0.53788 | 0.69908 |
| XGBoost reducido alternativo | 0.53961 | 0.71860 |

La **selección anidada** eligió Extra Trees geográfico en los cinco folds: RMSE **0.50338 t/ha**, MAE **0.35150 t/ha**, R² **0.64768**. Estas cifras reproducen la referencia experimental, pero no son la evaluación oficial de FIRA.

**Interpretación:** si el objetivo es estimar parcelas del mismo conjunto de regiones, la evidencia favorece provisionalmente Extra Trees geográfico. Si queremos extrapolar a municipios que no aparecen durante el entrenamiento, el XGBoost reducido conservador fue mejor en esta prueba. La selección final depende de la tarea real; no confundimos ambas evaluaciones.

## F. Prueba exploratoria adicional: combinar modelos

También evaluamos offline mezclas de las predicciones fuera de muestra de Extra Trees y XGBoost con ponderaciones fijas. Utilizar 10% XGBoost y 90% Extra Trees produjo RMSE medio aproximado de **0.49653**, frente a **0.49634** con Extra Trees solamente. La mezcla **no mejoró** en la prueba exploratoria. No se incorpora como ganador.

## G. Limitaciones para documentar ante FIRA

- Con 138 etiquetas, una pequeña diferencia puede no ser estable.
- Los rendimientos conocidos son de un solo ciclo agrícola de 2025; no probamos predicción generalizable entre años de cosecha.
- La selección de candidatos fue informada por experimentos anteriores; la anidada es rigurosa a nivel de folds, pero no totalmente independiente a nivel del proyecto.
- Validar en los mismos municipios se parece más al conjunto oculto del concurso; extrapolar a municipios desconocidos es una pregunta científica distinta.
- Los índices satelitales medidos durante el ciclo **no** permiten afirmar que hacemos una predicción antes de sembrar usando solo mes y ubicación.
- Los datos y código generados aquí no se subieron a GitHub; lo haremos con una versión estable y acordada por el equipo.

## H. Fuentes metodológicas para estudiar

- Documentación oficial de XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Validación cruzada anidada en scikit-learn: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
