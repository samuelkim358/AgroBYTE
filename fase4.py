"""Ejecutamos el cierre candidato de AgroBYTE con el intérprete activo."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import subprocess  # Lanzamos el módulo de ciencia de datos sin cambiar el entorno Python.
import sys  # Usamos el intérprete donde instalamos las dependencias.
from pathlib import Path  # Encontramos el proyecto independientemente de su ubicación.



# ==================== 2. INICIO DE LA FASE 4 ====================
if __name__ == '__main__':  # Ejecutamos el entrenamiento solamente cuando llamamos este archivo.
    raiz = Path(__file__).resolve().parent  # Localizamos la carpeta principal de AgroBYTE.
    comando = [sys.executable, str(raiz / 'codigo' / 'fase4_cierre_modelo.py'), *sys.argv[1:]]  # Pasamos cualquier opción adicional al módulo científico.
    raise SystemExit(subprocess.call(comando, cwd=raiz))  # Conservamos los mensajes y errores reales de la ejecución.
