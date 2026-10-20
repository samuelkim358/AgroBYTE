# Guía docente: cómo evaluamos modelos en AgroBYTE

## 1. Experimento sencillo con cuatro parcelas (valores ficticios)

| Parcela | Real (t/ha) | Predicción (t/ha) | Error (real − predicho) |
|---|---:|---:|---:|
| A | 2 | 2 | 0 |
| B | 3 | 3 | 0 |
| C | 4 | 4 | 0 |
| D | 5 | 7 | −2 |

- **MAE** = promedio del error absoluto = `(0+0+0+2)/4 = 0.50 t/ha`.
- **RMSE** = raíz del promedio del error al cuadrado = `√((0²+0²+0²+2²)/4) = 1.00 t/ha`. Castiga más los errores grandes.
- **R²** = `1 − Σ(yi−ŷi)² / Σ(yi−media(y))²`. La media real es 3.5, la suma de cuadrados de referencia es 5 y el R² es `1−4/5 = 0.20`.

`R² = 0.20` no significa acertar el 20% de las parcelas. Significa reducir un 20% la suma de errores cuadrados respecto a predecir la media del conjunto evaluado. R² puede ser negativo.

## 2. Cómo medimos errores sin conocer los 59 rendimientos ocultos

1. Apartamos algunas parcelas con rendimiento conocido antes de entrenar.
2. Entrenamos con las otras parcelas etiquetadas.
3. Predecimos las parcelas apartadas sin mostrar sus rendimientos al modelo.
4. Comparamos predicción contra rendimiento real y calculamos métricas.
5. Repetimos el procedimiento en otras particiones (*validación cruzada*).

**Regla de oro:** medimos MAE, RMSE y R² exclusivamente donde existe una etiqueta real de rendimiento. Las 59 parcelas de predicción se reservan para inferencia, no para evaluar precisión.

## 3. Parcelas frente a municipios

La validación por **parcelas** aproxima el problema de predecir otras parcelas de zonas representadas en entrenamiento. La validación por **municipios** pregunta si el modelo predice bien una zona completa sin ejemplos de esa zona. Son preguntas diferentes y sus resultados no deben mezclarse.

## 4. Errores que debemos evitar

- No entrenar con la parcela cuyo rendimiento estamos evaluando.
- No usar `ID_POLIGONO`, `CONJUNTO` ni `RENDIMIENTO_T_HA` como características predictoras.
- No calcular transformaciones supervisadas usando las etiquetas del conjunto de validación.
- No declarar que un R² de 0.65 implica 65% de aciertos.
- No considerar probados los umbrales de nubosidad o las mejoras que aún estamos explorando.
- No llamar oficiales a las predicciones marcadas `NO_ENTREGAR`.
