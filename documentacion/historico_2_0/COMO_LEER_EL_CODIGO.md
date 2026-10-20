# Cómo leer el código de AgroBYTE

Empezamos por `ejecutar.py`: recibe una orden breve, localiza el dataset y ejecuta la fase correspondiente. En `codigo/rutas.py` resolvemos automáticamente la carpeta `DATASET_RETO_AGRO_2026` sin rutas personales. Dentro de cada archivo `.py` encontraremos encabezados de sección y comentarios de operaciones formulados en primera persona plural implícita: **«importamos», «calculamos», «entrenamos»**.

## Orden recomendado de lectura

1. `codigo/rutas.py`: ubicación y protección de datos originales.
2. `codigo/experimento_agrocebada.py`: limpieza, agregación satelital, rásteres y experimentos iniciales.
3. `codigo/fase2_modelos_comparacion.py`: candidatos, preprocesamiento por pliegue, comparación y predicciones preliminares.
4. `codigo/diagnosticos_fase2.py`: comprobaciones y análisis de errores regionales.

Los `print()` enseñan resultados, las funciones reciben datos y devuelven transformaciones o estimaciones, y los CSV que escribimos aparecen exclusivamente bajo `salidas/`. Las líneas del código pueden ser largas porque conservamos trazabilidad de todas las operaciones del pipeline previo. No interpretamos un `assert` o una métrica como una garantía de precisión futura.

Para las matemáticas comenzamos con `METRICAS_Y_VALIDACION_EXPLICADAS.md`; para estudiar algoritmos, consultamos `MODELOS_EXPLICADOS_Y_USO_EN_AGROBYTE.md`.
