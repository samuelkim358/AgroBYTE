"""Punto de entrada adicional de la Fase 3; no modifica ejecutar.py ni las fases anteriores."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import subprocess  # Ejecutamos el módulo científico usando el mismo entorno de Python.
import sys  # Detectamos el intérprete activo del equipo.
from pathlib import Path  # Localizamos el proyecto sin escribir rutas locales de Windows.


# ==================== 2. EJECUCIÓN DE LA FASE 3 ====================
if __name__ == '__main__':  # Iniciamos el experimento únicamente desde la terminal.
    raiz = Path(__file__).resolve().parent  # Identificamos la carpeta que contiene este programa.
    comando = [sys.executable, str(raiz / 'codigo' / 'fase3_xgboost.py'), *sys.argv[1:]]  # Transferimos las opciones, incluyendo --rapido.
    raise SystemExit(subprocess.call(comando, cwd=raiz))  # Devolvemos el código real de salida para detectar problemas.
