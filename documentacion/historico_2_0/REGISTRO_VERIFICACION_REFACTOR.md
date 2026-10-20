# Registro de verificación técnica — Refactorización colaborativa

**Objetivo:** comprobar que mover el dataset dentro del proyecto y reemplazar rutas absolutas por rutas relativas **no modifica los algoritmos ni los resultados numéricos**.

## Pruebas realmente ejecutadas

1. **Compilación de sintaxis** de `ejecutar.py`, `codigo/*.py` y `tests/*.py`: aprobada.
2. **Pruebas automáticas** con `python -m unittest discover -s tests -v`: 3/3 superadas.
3. **Localización de insumos** usando `python ejecutar.py verificar` sobre una carpeta interna `DATASET_RETO_AGRO_2026`: detectó los CSV, componentes del Shapefile y ZIP necesarios.
4. **Fase 1 completa**: terminó correctamente, procesó 197 parcelas (138 etiquetadas y 59 ocultas) y reprodujo RMSE ≈ 0.496432 t/ha del modelo provisional Extra Trees geográfico.
5. **Fase 2 completa**: terminó correctamente, calculó CV repetida, CV espacial y CV anidada y guardó el modelo experimental; RMSE anidado ≈ 0.503384 t/ha.
6. **Diagnóstico posterior**: verificó exactamente 59 predicciones distintas, valor positivo finito y fórmula de producción total; generó métricas por estado.

Estos resultados confirman que el proyecto funciona en el **entorno de prueba** utilizado. No sustituyen las pruebas de instalación de dependencias en las computadoras de cada miembro ni validan todavía una aplicación web.

## Control de alcance

- La reorganización y documentación de comentarios no cambian el conjunto de ejemplos etiquetados ni hacen públicos los rendimientos ocultos.
- Los resultados de modelos son **provisionales**. No presentamos el ranking exploratorio como desempeño oficial de FIRA.
- La carpeta oficial se trata como fuente de lectura. Los archivos de salida se escriben en `salidas/`.
- La carpeta `DATASET_RETO_AGRO_2026` no está incluida en el ZIP por precaución sobre permisos de redistribución. Cada integrante debe disponer de copia autorizada y colocarla dentro del proyecto.

## Entorno usado para la prueba

- Python: **3.13.5**.
- Sistema: Linux en entorno de comprobación.
- Versiones de paquetes relevantes: 
  - `numpy`: `2.3.5`
  - `pandas`: `2.2.3`
  - `geopandas`: `1.1.2`
  - `rasterio`: `1.5.0`
  - `pyproj`: `3.7.2`
  - `shapely`: `2.1.2`
  - `scikit-learn`: `1.8.0`
  - `catboost`: `1.2.8`
  - `joblib`: `1.5.3`
  - `matplotlib`: `3.10.8`
  - `scipy`: `1.17.0`
