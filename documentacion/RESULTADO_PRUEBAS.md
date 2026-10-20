# Registro de pruebas previo a publicación

- **Origen de artefactos:** ZIP AgroBYTE PROJECT 2.0 entregado por el equipo el 15/10/2026. No modificamos pesos de Extra Trees ni las 59 estimaciones.
- **Firmas SHA-256:** modelo, entradas de 59 parcelas, CSV de predicciones y polígonos congelados en `datos/version_entrega.json`.
- **Reproducción del modelo:** `model.predict()` generó 59 rendimientos, cada uno dentro de 1e-6 t/ha del CSV auditado. En el entorno de comprobación sklearn 1.8.0 avisó diferencia de versión respecto al sklearn 1.9.1 original, pero las predicciones coincidieron.
- **Geometrías:** se verificaron 59 polígonos no duplicados, válidos, cuyos ID coinciden con las 59 filas de inferencia.
- **Controles de negocio:** superficie por rendimiento coincide con toneladas por parcela; se rechazan ID repetidos, montos negativos e inconsistencias de CSV.
- **Pruebas nuevas:** 10 aprobadas (`python -m unittest discover -s pruebas_v3 -v`).
- **Pruebas históricas del código científico:** 25 aprobadas en el entorno de revisión (`python -m unittest discover -s tests -v`). Para repetirlas, instalar `requirements_investigacion.txt`.
- **Compilación Python:** sintaxis de `app.py`, `verificar_entrega.py` y `codigo/entrega_v3.py` comprobada.
- **Limitación de verificación gráfica:** el entorno de construcción no pudo instalar Streamlit por falta de acceso al índice de paquetes, así que no se validó la apertura real del navegador o su modo oscuro. Esta prueba es **obligatoria** en una computadora del equipo antes de proporcionar el enlace a FIRA.
- **Despliegue:** pendiente; no se inventó URL pública ni video.

**Decisión:** paquete candidato listo para prueba local y revisión colaborativa, **no marcado como enviado a FIRA**.
