"""Rutas portables de AgroBYTE: los datos oficiales se leen y las salidas se aíslan."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from pathlib import Path  # Importamos Path para construir rutas que funcionen en Windows, Linux y macOS.



# ==================== 2. CARPETAS DEL PROYECTO ====================




# ==================== RAIZ PROYECTO ====================
def raiz_proyecto():  # Identificamos la carpeta principal tomando como referencia la ubicación de este archivo.
    return Path(__file__).resolve().parent.parent  # Regresamos a la raíz sin depender de la terminal ni del nombre de usuario.



# ==================== 3. DATOS ORIGINALES ====================




# ==================== CARPETA DATASET ====================
def carpeta_dataset(ruta_alternativa=None):  # Localizamos el dataset oficial dentro del proyecto, salvo que se indique otra ruta.
    if ruta_alternativa is not None:  # Permitimos una ubicación alternativa para pruebas sin cambiar el código fuente.
        return Path(ruta_alternativa).expanduser().resolve()  # Normalizamos la ruta opcional proporcionada explícitamente.
    return (raiz_proyecto() / 'DATASET_RETO_AGRO_2026').resolve()  # Buscamos el mismo nombre de carpeta en cualquier computadora.



# ==================== 4. PROTECCIÓN DE ARCHIVOS ORIGINALES ====================




# ==================== VALIDAR CARPETA SALIDA ====================
def validar_carpeta_salida(datos, salida):  # Impedimos que los experimentos sobrescriban o creen resultados en el dataset.
    origen = Path(datos).resolve()  # Obtenemos la ubicación real del conjunto oficial.
    destino = Path(salida).resolve()  # Obtenemos la ubicación real donde queremos escribir resultados.
    if destino == origen or origen in destino.parents:  # Detectamos un destino igual o dentro de los originales.
        raise ValueError('La carpeta de salida no puede estar dentro de DATASET_RETO_AGRO_2026.')  # Detenemos una configuración peligrosa.
    return destino  # Devolvemos la ubicación verificada para guardar resultados.
