# Decisión científica — candidato Extra Trees para las 59 parcelas

## Hipótesis y objetivo

La tarea del concurso consiste en estimar el rendimiento (t/ha) de **59 parcelas identificadas**, cuyas variables geográficas y series satelitales están disponibles. No pretendemos predecir parcelas arbitrarias de México ni estimar un cultivo antes de sembrar.

## Evidencia disponible

- Datos: 138 parcelas con rendimiento conocido y 59 sin rendimiento.
- Resultado por parcelas: Extra Trees geográfico presenta un RMSE exploratorio de aproximadamente 0.496 t/ha; CatBoost y las variantes XGBoost evaluadas no lo superaron bajo ese protocolo.
- Evaluación anidada: aproximadamente RMSE 0.503 t/ha, MAE 0.351 t/ha, R² 0.648. Extra Trees geográfico fue elegido en los cinco pliegues externos.
- Evaluación por municipios no vistos: XGBoost rinde mejor en el protocolo espacial; por tanto, no afirmamos que Extra Trees sea universalmente superior.
- Auditoría de cobertura: las 59 parcelas pertenecen a municipios con entrenamiento; 10 centroides quedan fuera de la envolvente convexa de su estado. La distancia mediana al ejemplo etiquetado más cercano es aproximadamente 350 m.
- Limitación agronómica: cuando el rendimiento es excepcionalmente bajo, Extra Trees puede sobreestimarlo; cuando es alto, puede subestimarlo.

## Regla de elección

**Se selecciona Extra Trees como candidato preferido para la integración local**, debido al rendimiento por parcelas y la representación geográfica de las 59. La elección es condicional al conjunto, las divisiones y las familias ensayadas; no prueba una precisión oficial ni óptimo global.

La interfaz lee las métricas locales y `auditar_modelo_v2.py` genera un dictamen desde el resultado realmente reproducido en esa computadora. Si cambian los archivos o la configuración, debe regenerarse el dictamen.

## Modelo verdadero y alcance de la interfaz

El candidato utiliza exactamente las variables `AREA_HA`, `latitud`, `longitud`, `estado` y `municipio`. Los índices satelitales se evaluaron en modelos alternativos, pero **no son entradas del predictor seleccionado**. La interfaz **no vuelve a entrenar ni predice nuevos terrenos**; solo consulta el CSV revisado.

## Antes de entregar a FIRA

Verificar la integración del dashboard, los 59 IDs y las unidades, la firma de los artefactos, el esquema exacto esperado de entrega, la documentación del modelo y de la IA utilizada, el enlace accesible para el jurado y la revisión conjunta de supuestos/limitaciones.
