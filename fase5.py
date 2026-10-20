"""Iniciamos el análisis complementario de FIRA sin reentrenar nuestro modelo candidato."""


# ==================== 1. IMPORTAMOS LIBRERÍAS ====================
import subprocess  # Ejecutamos la fase auxiliar con el mismo Python del equipo.
import sys  # Recuperamos el intérprete del entorno activo.
from pathlib import Path  # Localizamos archivos sin rutas absolutas personales.


# ==================== 2. INICIAMOS LA FASE 5 ====================
RAIZ = Path(__file__).resolve().parent  # Ubicamos la raíz compartida del proyecto.
if __name__ == '__main__':  # Comprobamos que se pidió ejecutar este archivo directamente.
    subprocess.run([sys.executable,str(RAIZ/'codigo'/'fase5_indices_clima.py'),*sys.argv[1:]],cwd=RAIZ,check=True)  # Lanzamos el análisis y propagamos errores reales.
    subprocess.run([sys.executable,str(RAIZ/'codigo'/'graficas_fase5.py')],cwd=RAIZ,check=True)  # Dibujamos las figuras en otro proceso para liberar la memoria usada por los modelos.
