# AgroBYTE — Revisión geográfica final del candidato Extra Trees

## Contexto y pregunta

**Pregunta:** ¿Las 59 parcelas ocultas se encuentran en regiones con suficiente representación geográfica dentro de las 138 parcelas etiquetadas para que nuestro candidato Extra Trees geográfico sea razonable?

**Fuentes utilizadas:** CSV oficial de rendimientos, Shapefile de FIRA, CSV candidato de la Fase 4 y predicciones fuera de muestra de la validación anidada de la Fase 3. El rendimiento real de las 59 parcelas ocultas **no está disponible ni se ha utilizado**.

## Hallazgos medidos

| Indicador | Resultado |
|---|---:|
| Parcelas etiquetadas / ocultas | 138 / 59 |
| Municipios de parcelas ocultas con ejemplos etiquetados | 59 de 59 |
| Parcela etiquetada más cercana en el mismo municipio | 58 de 59 |
| Distancia mediana al centro de la parcela etiquetada más cercana | 0.350 km |
| Distancia máxima al centro de la parcela etiquetada más cercana | 1.923 km |
| Percentil 95 de la distancia entre parcelas etiquetadas distintas | 2.027 km |
| Parcelas ocultas por encima de ese umbral | 0 de 59 |
| Parcelas ocultas dentro de la envolvente convexa estatal de los centros etiquetados | 49 de 59 |
| Parcelas ocultas fuera de esa envolvente | 10 de 59 |
| Predicciones en el intervalo de rendimientos observados de su estado | 59 de 59 |

**Matiz:** 10 parcelas quedan fuera de la envolvente geométrica estatal, aunque tienen vecinos próximos. No son casos de extrapolación a municipios completamente desconocidos, pero tampoco son estrictamente interiores a la envolvente.

## Cobertura por estado

| Estado | Ocultas | Distancia mediana al vecino (km) | Distancia máxima (km) | Fuera de la envolvente estatal |
|---|---:|---:|---:|---:|
| Hidalgo | 15 | 0.448 | 1.923 | 4 |
| Puebla | 29 | 0.290 | 0.839 | 3 |
| Tlaxcala | 15 | 0.575 | 1.702 | 3 |

## Cuatro parcelas que merecen revisión individual

| ID | Municipio | Distancia al centro del vecino conocido |
|---|---|---:|
| AGC_065 | Almoloya | 1.923 km |
| AGC_173 | Calpulalpan | 1.702 km |
| AGC_174 | Cuautepec de Hinojosa | 1.645 km |
| AGC_120 | Nanacamilpa de Mariano Arista | 1.265 km |

La AGC_120 tiene un municipio representado por una parcela de entrenamiento, pero su vecino etiquetado más cercano se ubica en Calpulalpan. Esto no invalida la predicción; señala una cobertura local menos abundante.

## Sesgo fuera de muestra del modelo candidato

En la validación anidada sobre las 138 parcelas etiquetadas, las observaciones de rendimiento más bajo (2.0–3.2 t/ha) fueron, en promedio, **sobreestimadas 0.397 t/ha**. Las más altas (4.55–5.59 t/ha) fueron, en promedio, **subestimadas 0.400 t/ha**. Este comportamiento es consistente con un modelo de árboles promediados que suaviza extremos regionales, pero no prueba que el mismo sesgo se produzca en cada parcela oculta.

La parcela **AGC_171 (Puebla)** sirve como advertencia: su rendimiento real conocido fue 2.15 t/ha, y la predicción fuera de muestra fue aproximadamente 4.47 t/ha. Aunque el RMSE global es relativamente bajo, algunos errores individuales fueron grandes.

## Decisión técnica propuesta

**Extra Trees geográfico sigue siendo el modelo candidato recomendado para esta entrega concreta**, porque ganó la validación por parcelas en los experimentos realizados y las 59 ubicaciones ocultas tienen una cobertura geográfica comparable a la muestra etiquetada. **No afirmamos que sea el óptimo global ni que garantice acierto parcela por parcela.**

Antes de aprobar el CSV como entrega definitiva, el equipo debe confirmar el formato requerido, revisar las 10 ubicaciones exteriores, verificar hashes de datos/modelo y revisar que la aplicación interprete `RENDIMIENTO_PREDICHO_T_HA` como t/ha y `PRODUCCION_ESTIMADA_T` como toneladas totales.

Las pruebas espaciales por municipios nuevos siguen siendo relevantes para el futuro producto, pero no deben confundirse con la tarea de predicción de las 59 parcelas del concurso.

## Trazabilidad

La auditoría produce `auditoria_59_parcelas.csv`, `auditoria_resumen_por_estado.csv`, `resumen_auditoria.json` y figuras; **no cambia el modelo `.joblib`, los datos oficiales ni las predicciones de la Fase 4**. Los archivos del código incluyen comentarios explicativos en primera persona plural implícita.
