# 🌾 AgroBYTE 3.0 — Reto AgroCebada FIRA 2026

**Prototipo académico web** para estimar y consultar el rendimiento de 59 parcelas oficiales de cebada de Puebla, Hidalgo y Tlaxcala, visualizar polígonos, ejecutar el modelo **Extra Trees** seleccionado, simular escenarios económicos editables y explicar resultados científicos. **No es un portal oficial de FIRA ni una herramienta de aprobación crediticia.**

## 1. Lo que contiene esta entrega

- `app.py`: interfaz Streamlit con **modo claro / oscuro** y seis secciones.
- `modelo/modelo_extratrees_geografico.joblib`: pipeline entrenado con las 138 parcelas etiquetadas de 2025.
- `datos/entradas_59.csv`: cinco variables requeridas para ejecutar el modelo sobre las 59 parcelas del reto.
- `datos/predicciones_59.csv`: predicciones candidatas auditadas (ID, estado, municipio, ha, t/ha, t).
- `datos/poligonos_59.geojson`: geometrías derivadas para cartografía; **confirmar autorización de exposición pública**.
- `datos/version_entrega.json`: firmas criptográficas SHA-256 del modelo y sus entradas.
- `codigo/entrega_v3.py`: validaciones, inferencia real e indicadores económicos comprobables.
- `evidencia/`: métricas de las Fases 3 y 5, importancia de índices y clima, auditoría geográfica.
- `codigo/` y `tests/`: scripts históricos de experimentación y pruebas del producto.
- `documentacion/`: metodología, borrador del informe, registro de uso de IA, requisitos, guía de despliegue y video.

**El dataset original de FIRA no se incluye.** El jurado puede probar las 59 inferencias de la aplicación sin tener que instalar 150 MB de datos originales ni reentrenar modelos. Para reproducir las fases históricas 1–5 sí se requiere colocar el dataset autorizado en `DATASET_RETO_AGRO_2026/` e instalar `requirements_investigacion.txt`.

## 2. Ejecución local recomendada

Instalar **Python 3.14**, idealmente la misma versión de Python y **scikit-learn 1.9.1** documentadas durante el entrenamiento. Se requiere internet **la primera vez** para instalar bibliotecas y para descargar teselas del mapa.

**Windows:** doble clic en `INICIAR_AGROBYTE_WINDOWS.bat`. La primera ejecución crea `.venv`, instala dependencias, comprueba las 59 predicciones y abre la interfaz.

**PowerShell manual:**

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verificar_entrega.py
.\.venv\Scripts\python.exe -m unittest discover -s pruebas_v3 -v
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:** `chmod +x INICIAR_AGROBYTE_MAC_LINUX.sh` y `./INICIAR_AGROBYTE_MAC_LINUX.sh`.

Abrimos `http://localhost:8501` cuando arranque Streamlit. Para detener, `Ctrl+C` en la terminal. Si `py -3.14` no está disponible, utilizamos `python -m venv .venv`, pero verificamos las 59 predicciones antes de iniciar.

## 3. ¿Cómo utiliza realmente el modelo?

1. Al iniciar, comprobamos los hashes SHA-256 de los archivos firmados.
2. Cargamos únicamente el `.joblib` creado por AgroBYTE (no aceptamos modelos externos por carga de usuario).
3. Ejecutamos **las 59 inferencias** con `AREA_HA`, `latitud`, `longitud`, `estado` y `municipio`.
4. Verificamos que el nuevo resultado coincide con el CSV de la Fase 4 (tolerancia 0.000001 t/ha).
5. Al pulsar **Calcular rendimiento**, ejecutamos `predict()` para esa parcela oficial. No reentrenamos el modelo.
6. Producción total = hectáreas × rendimiento predicho (t/ha).

Si cambian los artefactos o la serialización ya no reproduce los resultados, **la aplicación se detiene**, no inventa cifras.

## 4. Funciones principales

- **Resumen:** KPIs por las 59 parcelas y resultados agregados por estado.
- **Mapa y parcelas:** polígonos reales de las 59 parcelas, filtros y tabla de predicciones.
- **Ejecutar modelo:** predicción individual en tiempo real con el pipeline guardado.
- **Simulador económico:** precio y costos editables, margen, punto de equilibrio, sensibilidad ±15%, interés simple opcional.
- **Ciencia de datos:** RMSE, MAE y R² de validación, comparaciones, LAI/EVI, precipitación CHIRPS.
- **Documentación y descargas:** exportar CSV filtrado o 59 pares ID-rendimiento y consultar trazabilidad.

**Advertencia:** no se garantiza predicción para terrenos arbitrarios, meses distintos o una siembra futura. Los índices satelitales y climáticos del ciclo 2025 se estudiaron retrospectivamente; el ganador geográfico NO usa esos índices como entradas directas.

## 5. Pruebas y reproducibilidad

```powershell
python verificar_entrega.py
python -m unittest discover -s pruebas_v3 -v
```

Las pruebas comprueban 59 IDs únicos, predicciones válidas, integridad criptográfica, coherencia de unidades, inferencia y cálculo económico. El código histórico queda en `codigo/`, separado del visor. El artefacto se entrenó originalmente con **Python 3.14.7 / scikit-learn 1.9.1**, según manifiesto adjunto; cargas en otra versión no son garantía de compatibilidad, aunque las predicciones deben superar el chequeo estricto.

## 6. Publicación para FIRA — pasos pendientes del equipo

1. **Confirmar autorización para exhibir en internet los polígonos geográficos** derivados del dataset proporcionado. Si no está autorizado, debemos retirar `datos/poligonos_59.geojson` y ofrecer solo una vista no geográfica o privada; el modelo puede seguir funcionando con las 59 filas.
2. Crear el **repositorio de entrega** y publicar una versión revisada, no publicar el dataset original ni credenciales.
3. Desplegar Streamlit (véase `documentacion/DESPLIEGUE_FIRA.md`) y **probar la URL desde un equipo externo** sin credenciales.
4. Grabar y subir un video de **máximo cinco minutos**; conservar su URL real.
5. Revisar los **textos de informe** con sus límites de palabras en la plataforma y completar aportaciones reales y prompts usados.
6. Comprobar el CSV, fuentes, firma, anexos y video; únicamente entonces pulsar **Publicar** en FIRA antes del **19 de octubre de 2026, 23:59 h**.

**La entrega no puede considerarse enviada hasta que el equipo publique sus enlaces y anexos en la plataforma oficial.**

## 7. Limitaciones y honestidad científica

RMSE anidado ≈ **0.5034 t/ha**, MAE ≈ **0.3515 t/ha**, R² ≈ **0.6477** provienen de validación sobre 138 etiquetas conocidas, **no** de la evaluación oculta de FIRA. El modelo puede sobreestimar rendimientos bajos y subestimar los extremos altos, y se generaliza peor a municipios nuevos. La simulación financiera es ilustrativa y no constituye dictamen crediticio.
