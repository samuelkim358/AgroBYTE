# Guía de matemáticas y experimentación — AgroCebada (nivel estudiante)

## 1. Qué estamos aprendiendo

Cada parcela tiene variables de entrada X (latitud, longitud, hectáreas, índices de vegetación, precipitación, temperatura, elevación y otras). Las 138 parcelas de entrenamiento también tienen un valor `RENDIMIENTO_T_HA`. Buscamos una función `f(X)` que se aproxime al rendimiento `y` de parcelas nuevas.

`predicción = f(variables de la parcela)`; el programa no dispone de los 59 rendimientos reales, por lo que no puede consultarlos ni utilizarlos para verificar resultados.

## 2. Una prueba simple: ocultar un rendimiento conocido

Si la parcela A rindió 4.50 t/ha y el modelo predijo 4.20 t/ha, la diferencia es `4.20 - 4.50 = -0.30 t/ha`, el error absoluto es `|-0.30| = 0.30 t/ha` y el cuadrado del error es `(-0.30)^2 = 0.09`.

`MAE = (1/n) * sum(|y_i - pred_i|)` se interpreta como error absoluto promedio en toneladas por hectárea.

`RMSE = sqrt[(1/n) * sum((y_i - pred_i)^2)]` da más importancia a errores grandes y también se expresa en t/ha.

`R² = 1 - sum((y_i-pred_i)^2) / sum((y_i-media_y)^2)` mide mejora frente a predecir la media de la muestra evaluada; puede ser negativo.

**No sería una prueba válida** medir el error de las mismas parcelas que el modelo usó para aprender. Necesitamos separar conjuntos y repetir el proceso.

## 3. ¿Qué significa validar cinco veces?

Dividimos los 138 ejemplos en cinco grupos. Entrenamos con cuatro y predecimos el quinto. Lo repetimos hasta que cada parcela haya sido evaluada sin haber participado en el entrenamiento de su propia predicción. Con cada semilla, obtenemos 138 predicciones *out of fold* (OOF). Repetimos con semillas 42, 73 y 2026 para estudiar la estabilidad de los resultados. Por eso hay 15 entrenamientos de cada configuración.

Las mismas 138 parcelas aparecen en todas las semillas, así que **no tenemos 414 parcelas independientes**. La desviación entre semillas tampoco es un intervalo de confianza estadístico.

## 4. ¿Qué aprende Extra Trees?

Extra Trees construye un conjunto de árboles de decisión. Cada árbol divide los ejemplos mediante reglas sobre las características, por ejemplo `latitud < 19.6` (ejemplo ilustrativo, NO regla demostrada del modelo). Las divisiones se seleccionan entre candidatos aleatorizados. Cada hoja estima el rendimiento a partir de ejemplos que llegan a ella; el bosque promedia los resultados de sus árboles. Esto reduce la variación respecto a utilizar un árbol único, aunque puede seguir sobreajustándose si es demasiado complejo.

En nuestro experimento usamos `150` árboles, `min_samples_leaf=3` y `max_features=0.85`. Con solo 138 parcelas conocidas evitamos árboles sin restricciones.

## 5. ¿Qué aprende CatBoost?

CatBoost es una forma de boosting: combina sucesivamente árboles relativamente pequeños, de modo que cada uno intente mejorar el error del conjunto anterior. Una representación conceptual es `F_m(X) = F_(m-1)(X) + tasa_aprendizaje * arbol_m(X)`.

Usamos profundidad 3, 350 iteraciones, tasa de aprendizaje 0.035 y regularización L2 en las hojas. Estos parámetros son hipótesis comparativas: no afirmamos que sean óptimos para FIRA. La implementación de este paquete utiliza una transformación numérica previa; **no explota directamente el manejo nativo de categorías de CatBoost**.

## 6. ¿Para qué sirve Ridge?

Una regresión lineal construye `y_pred = beta_0 + beta_1*X_1 + ... + beta_p*X_p`. Ridge minimiza un error cuadrático junto con una penalización de parámetros, de forma conceptual `sum((y - y_pred)^2) + lambda*sum(beta_j^2)`.

La penalización dificulta que los coeficientes se vuelvan excesivos cuando hay muchas variables correlacionadas. Las variables numéricas se normalizan en el pipeline, y la normalización se ajusta únicamente con datos de cada entrenamiento.

## 7. ¿Por qué usamos una segunda validación por municipios?

La validación habitual por parcela responde: *¿predigo parcelas desconocidas en regiones donde conozco otras parcelas cercanas?*.

La validación agrupada por municipio responde una pregunta más difícil: *¿qué pasa si no he visto ningún ejemplo de un municipio durante el entrenamiento?*.

Los errores más altos en ese escenario nos advierten que la ubicación ayuda a interpolar, pero no necesariamente a generalizar. Puebla tiene solo un municipio representado: no podemos evaluar generalización dentro de Puebla con los datos actuales.

## 8. ¿Por qué hacemos validación cruzada anidada?

Si probamos siete modelos y escogemos el que obtuvo menor error en las mismas parcelas, ese valor podría estar favorecido por la búsqueda. En la validación anidada, cada fold **externo** se guarda como una prueba de la selección hecha únicamente dentro del resto de los datos, mediante folds **internos**.

Aquí evaluamos tres familias predefinidas para esa ejecución anidada: Extra Trees geográfico, Extra Trees completo y CatBoost reducido. El RMSE externo fue 0.5034 t/ha. **Reserva importante:** estas familias se eligieron después de exploraciones previas de las mismas 138 etiquetas, por lo que no consideramos el resultado un test totalmente independiente de todo el proyecto.

## 9. ¿Por qué no utilizar todavía los 59 rendimientos ocultos?

Porque esos rendimientos no aparecen en nuestros datos y FIRA los utilizará para evaluar la solución. Sí podemos generar 59 predicciones, comprobar que no faltan IDs y que las toneladas totales se calculen correctamente. No podemos decir cuántas acertamos hasta conocer sus rendimientos reales.

## 10. Fórmula para el dashboard financiero

Si la predicción es 4.2 toneladas por hectárea y la parcela ocupa 3.5 hectáreas, la producción estimada es `4.2 * 3.5 = 14.7 toneladas`.

Si el productor introduce un precio de `$5,000 MXN/tonelada`, el ingreso bruto estimado es `$73,500 MXN`; si reporta costos estimados por `$40,000`, la utilidad estimada sería `$33,500`. Estos números son únicamente ejemplos didácticos, no precios reales ni predicciones AgroCebada.
