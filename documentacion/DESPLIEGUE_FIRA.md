# Despliegue y revisión final del prototipo

## Antes de publicar

Confirmamos autorización para mostrar polígonos y coordenadas derivados de las parcelas del dataset original. El paquete no incluye originales, pero sí derivados; el equipo es responsable de cumplir las condiciones de FIRA. Si no está permitido compartir esos derivados en público, debemos **retirar el geojson** y cambiar el mapa por un resumen regional sin coordenadas expuestas. No ejecutamos sin autorización un repositorio público del dataset.

## Con GitHub y Streamlit Community Cloud

1. Subimos **la carpeta raíz completa** del proyecto a un repositorio de entrega. No subimos `.venv`, `DATASET_RETO_AGRO_2026` ni credenciales.
2. Verificamos que `app.py`, `requirements.txt`, `.streamlit/config.toml`, `modelo/`, `datos/` y `evidencia/` sean accesibles desde ese repositorio.
3. En Streamlit Community Cloud elegimos el repositorio, rama, y archivo de entrada `app.py`. Elegimos una versión de Python compatible con el artefacto (entrenado con Python 3.14.7 y scikit-learn 1.9.1). Si la plataforma no ofrece la misma versión, **no damos por válido el despliegue sin pasar `verificar_entrega.py`**.
4. Revisamos los logs del despliegue; si falta una dependencia, corregimos `requirements.txt` y reproducimos.
5. Abrimos la URL pública desde una ventana privada y desde una computadora de alguien que no esté conectado a GitHub. El jurado debe poder consultar el modelo y descargar el CSV sin solicitar permisos personales.
6. Ejecutamos una inferencia en «Ejecutar modelo» y contrastamos el rendimiento con `datos/predicciones_59.csv`.
7. Contrastamos estados, superficies, mapa, exportación de CSV y calculadora. Hacemos una captura como evidencia.
8. Comprobamos fecha límite y pegamos la **URL real** en el campo «Enlaces» de la plataforma FIRA.

Si el hosting elegido no soporta la versión exacta de Python/scikit-learn, utilizamos un entorno controlado propio (por ejemplo contenedor Docker compatible) antes que sustituir silenciosamente el modelo. Nunca mostramos un CSV precalculado como si el modelo hubiera ejecutado inferencia cuando no puede cargarse.

## Archivos que no son el enlace

`http://localhost:8501` funciona solo en el equipo del desarrollador y **no sirve** como enlace público del jurado. No enviamos un link inexistente ni asumimos que la plataforma creó uno.

**Nota sobre Python:** en Streamlit Community Cloud no debemos depender de `runtime.txt` para seleccionar la versión: la opción oficial es **Advanced settings → Python version** durante la creación del despliegue.
