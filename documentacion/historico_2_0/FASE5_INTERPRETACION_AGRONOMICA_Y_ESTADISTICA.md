# Fase 5 · Interpretación agronómica y límites estadísticos

## Procedencia de las variables

El diccionario de FIRA caracteriza LAI como **índice de área foliar**, EVI como **índice mejorado de vegetación**, NDVI como indicador de **vigor y actividad de la vegetación verde**, y MSAVI como índice de vegetación ajustado a la influencia del suelo. Estas definiciones describen el propósito de la medición, no prueban que un índice explique causalmente un rendimiento mayor.

Las mediciones del Dataset Básico incluyen Sentinel-2 y Landsat; el PRO incluye Planet. Aunque tengan el mismo nombre (p. ej., NDVI de Sentinel-2 y NDVI de Planet), **no debemos tratarlas como equivalentes automáticamente**. Por eso se muestran importancias por familia y también por proveedor.

La lluvia CHIRPS está en mm mensuales acumulados; CHIRTS-ERA5 incluye temperaturas mínima y máxima mensuales; las variables topográficas proceden de INEGI CEM4. Para el año 2025 se agregaron meses de abril a octubre y para 2022–2024 se incluyeron promedios históricos.

## Qué representa la fórmula de permutación

Si `RMSE_base` es el error del modelo sobre un conjunto que quedó fuera de entrenamiento y `RMSE_permutado(G)` el error después de romper la correspondencia de todas las columnas del grupo G, definimos:

`Importancia(G) = RMSE_permutado(G) − RMSE_base`.

- Valor **positivo**: el grupo contribuía a la predicción del modelo evaluado.
- Cerca de **cero**: poca dependencia detectable, redundancia con otras variables o efectos cancelados.
- Valor **negativo**: el modelo puede estar usando ruido o la perturbación puede beneficiarlo circunstancialmente. No equivale a una relación biológica negativa.

Importancia agrupada no significa que cada una de sus columnas aporte lo mismo. LAI y EVI aparecen en dos sensores (18 columnas por índice), mientras VI6T y MSAVI están en uno (9 columnas cada uno). Esta diferencia de cobertura puede influir en la puntuación. La tabla `importancia_indices_por_sensor.csv` separa estos efectos. Además, NDVI, EVI, SAVI y LAI pueden estar correlacionados; la importancia por permutación depende del resto de variables incluidas.

## Regresión y validación

Todos los modelos auxiliares usan el mismo `ExtraTreesRegressor` de Fase 2 (`n_estimators=150`, `min_samples_leaf=3`, `max_features=0.85`, `random_state=42`). Cambian **solo los grupos de variables**. Usamos cinco folds estratificados por estado y tres semillas para comparar RMSE. Esto permite comparar con el RMSE de la Fase 3, pero **no equivale a una prueba prospectiva independiente** después de tantas decisiones exploratorias.

Los modelos de clima pueden reconstruir parcialmente la geografía porque los píxeles raster y valores meteorológicos reflejan ubicación; no podemos atribuir su rendimiento únicamente a mecanismos agronómicos.

## Riesgo temporal

Los índices de abril–octubre 2025 y los resúmenes climáticos de ese mismo ciclo pueden estar disponibles solo **durante o después** de la temporada agrícola. Eso no implica fuga directa de la etiqueta de rendimiento, pero sí impide presentar estos modelos auxiliares como pronósticos hechos antes de la siembra de 2025. Para pronóstico prospectivo hacen falta datos conocidos antes de la fecha de predicción, seguimiento multitemporal y evaluación en otros ciclos agrícolas.

## Qué decisión comunica la ciencia

Para las 59 parcelas de FIRA, conservamos el modelo Extra Trees geográfico, porque tuvo menor RMSE en la evaluación por parcelas. Esta conclusión está limitada a la muestra, la estrategia de validación y las configuraciones ensayadas. Las preguntas guía se responden con evidencia complementaria; no se sustituyen las predicciones oficiales por un modelo de interpretación.
