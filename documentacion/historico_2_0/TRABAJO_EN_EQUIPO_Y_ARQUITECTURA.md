# Organización del repositorio y colaboración

## Objetivo común

Construir un predictor de rendimiento en t/ha a partir de fuentes FIRA y una aplicación que permita consultar 59 predicciones, mostrar parcelas y exportar un CSV. No utilizamos la aplicación para inventar un rendimiento que todavía no puede calcularse con datos adecuados.

## Responsabilidades propuestas

| Participante | Módulo | Entregable verificable |
|---|---|---|
| Ciencia de Datos | `codigo/` y experimentos | Pipeline, comparación, métricas, modelo validado y tabla de 59 predicciones |
| Desarrollo A | `app/paginas/dashboard.py` (futuro) | Dashboard con selección de parcela, rendimiento t/ha y toneladas |
| Desarrollo B | `app/paginas/mapa.py` (futuro) | Visualizador geográfico y cálculo de hectáreas |
| Desarrollo C | Integración, exportación, pruebas y despliegue | `st.download_button` de CSV, pruebas, enlace accesible y guía de uso |

**No hemos creado todavía los archivos de `app/`: son contratos de integración propuestos.** Quien implemente un componente debe probarlo y registrar qué cambios realizó.

## Contrato de predicción para la app

Los módulos deben aceptar un DataFrame con `ID_POLIGONO`, `estado`, `municipio`, `AREA_HA`, `RENDIMIENTO_PREDICHO_T_HA` y `PRODUCCION_ESTIMADA_T`. Esta última representa `AREA_HA * RENDIMIENTO_PREDICHO_T_HA` y se expresa en toneladas. Los valores de prueba deben marcarse como ficticios; los CSV `NO_ENTREGAR` son experimentales.

## Trabajo con GitHub

- Crear ramas por módulo: `feature/dashboard`, `feature/mapa`, `feature/exportacion`.
- Incluir una descripción de cambios, evidencia de pruebas y capturas en cada Pull Request.
- Evitar modificar el pipeline de Machine Learning a la vez que otros cambios sin coordinarse.
- El `.gitignore` excluye temporalmente archivos de datos oficiales y salidas generadas. Un clon del repositorio requiere copiar localmente `DATASET_RETO_AGRO_2026` con el nombre exacto.
- La carpeta original del reto puede contener datos cuya redistribución esté restringida. Confirmar permisos antes de compartirla o subirla a GitHub, incluso a un repositorio privado.

## Si el dataset ya fue agregado a Git

Después de crear el `.gitignore`, Git puede seguir rastreando archivos previamente versionados. Solo si el equipo decidió **no redistribuir el dataset** y desea retirarlo del próximo commit, podemos ejecutar desde la raíz:

```bash
git rm -r --cached DATASET_RETO_AGRO_2026
git status
```

La opción `--cached` elimina esos archivos del índice de Git, **no de nuestra carpeta local**. Esto no borra automáticamente copias ya almacenadas en el historial remoto; cualquier eliminación completa del historial requiere planificación adicional y revisión de permisos.
