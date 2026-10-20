# Bitácora científica — AgroCebada 2026 / Fase 2

## Experimento F2-01 — Comparación de modelos

**Pregunta:** ¿Un modelo espacial sencillo predice mejor que modelos complejos que combinan geografía, clima y series satelitales?

**Unidades y procedencia:** 197 parcelas con ID y geometría en fuentes FIRA; 138 etiquetas `RENDIMIENTO_T_HA`, 59 ocultas. Las matrices de fase 1 se construyeron agregando índices de abril a octubre de 2025, filtrando observaciones con nubosidad <=20% y extrayendo rásteres mensuales/estáticos por centroide. Tres grupos de variables: geografía (5), reducido (133), completo (166). Sin `ID_POLIGONO`, `CONJUNTO` ni rendimiento entre predictores.

**Método:** Siete combinaciones, cada una evaluada con 5 folds estratificados por estado y semillas 42, 73 y 2026. `Pipeline` aplica imputación por mediana, codificación categórica y, cuando corresponde, escala exclusivamente en el entrenamiento de cada fold. `ExtraTreesRegressor`, `CatBoostRegressor`, `HistGradientBoostingRegressor` y Ridge. Cada configuración tiene predicciones OOF auditables.

**Resultados:** Extra Trees geografía RMSE medio 0.4963 ± 0.0092 entre las tres semillas; completo Extra Trees 0.5118, reducido Extra Trees 0.5170, CatBoost reducido 0.5402 y CatBoost completo 0.5412. Esta desviación mide variabilidad descriptiva entre semillas, NO incertidumbre de la población.

**Conclusión provisional:** añadir más características no mejora automáticamente el RMSE de parcelas. No interpretar esto como prueba de falta de relevancia agronómica de los índices.

## Experimento F2-02 — Generalización espacial

**Pregunta:** ¿Qué ocurre al evaluar municipios completos excluidos del entrenamiento?

**Método:** `GroupKFold(n_splits=5)` usando `estado/municipio` como agrupación, mismo tratamiento de faltantes y mismos algoritmos.

**Resultados:** geográfico Extra Trees RMSE 0.9205; completo Extra Trees 0.8434; completo CatBoost 0.7811; reducido CatBoost 0.7725. El mejor resultado espacial corresponde a una pregunta estadística distinta; no inferir por ello que el modelo generalizará sin problemas a zonas del país no representadas.

**Limitaciones:** solo nueve municipios, distribución muy desigual y agrupaciones pequeñas. En Puebla únicamente hay ejemplos de Chignahuapan.

## Experimento F2-03 — Selección con validación anidada

**Método:** 5 folds externos (`random_state=319`) y, dentro de cada entrenamiento externo, 3 folds internos. Tres configuraciones fijas para esta ejecución: Extra Trees geográfico, Extra Trees completo y CatBoost reducido. Selección según RMSE interno; evaluación contra el fold externo separado. Resultados detallados en los CSV.

**Resultados:** 5/5 folds escogieron Extra Trees geográfico; RMSE externo 0.5034, MAE externo 0.3515, R² externo 0.6477. No utilizar estas cifras como métrica oficial de las 59 parcelas ocultas. Las familias elegidas ya habían sido consideradas en experimentos anteriores con los mismos datos; esta prueba reduce optimismo interno, pero no borra decisiones previas del proceso investigador.

## Experimento F2-04 — Diagnósticos y mezclas

**Auditoría de 59 predicciones:** exactamente 59 IDs únicos, igualdad exacta con el conjunto FIRA, valores finitos y positivos, y coherencia entre `AREA_HA * RENDIMIENTO_PREDICHO_T_HA` y `PRODUCCION_ESTIMADA_T`.

**Resultados por estado en CV anidada:** Hidalgo RMSE 0.3675 (36 parcelas), Puebla 0.5829 (71), Tlaxcala 0.4392 (31). Las diferencias pueden reflejar dificultad predictiva y composición distinta entre regiones; no son medida de causalidad.

**Mezcla posterior 75% geografía + 25% Extra Trees completo:** RMSE descriptivo 0.4947 frente a 0.4963 del geográfico puro. La diferencia es mínima y la mezcla se exploró después de ver las predicciones OOF. **No seleccionar mezcla como definitiva basándose en este mismo resultado sin validación anidada/nueva o estudio de estabilidad adicional.**

## Decisiones pendientes antes del modelo definitivo

1. Verificar importancia de índices, condiciones del cultivo y sesgo geográfico con explicabilidad fuera de muestra; estudiar selección de variables fenológicas y cambios de filtro de nubes mediante validación consistente.
2. Comprobar fuentes externas históricas de 2022-2024 de rendimiento municipal de cebada y posibles propiedades del suelo, sin agregar información de 2025 que codifique indirectamente la respuesta de ese mismo ciclo.
3. Congelar especificación única del modelo, versiones de bibliotecas, umbral de nubes, rasters y semillas antes de generar archivo oficial de las 59 parcelas.
4. Elaborar prototipo Streamlit, tabla/detalle por parcela, exportación CSV, pruebas funcionales en otra máquina y publicación con enlace verificable.
5. Registrar decisiones finales, evidencias, presentación, vídeo de cinco minutos  respetando límites del informe y anexos.
