"""Creamos visualizaciones de los experimentos ya exportados sin volver a entrenar."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from pathlib import Path  # Identificamos las salidas del proyecto.
import pandas as pd  # Leemos únicamente los CSV consolidados.
from fase5_indices_clima import guardar_graficas  # Reutilizamos las reglas de presentación.
from rutas import raiz_proyecto  # Localizamos el proyecto en cualquier máquina.


# ==================== 2. CONSTRUCCIÓN DE FIGURAS ====================
if __name__ == '__main__':  # Permitimos generar figuras sin reentrenar los modelos.
    carpeta = raiz_proyecto()/'salidas'/'fase5'  # Buscamos los resultados de la Fase 5.
    archivos = ['ranking_ablaciones.csv','importancia_indices.csv','importancia_indices_con_geografia.csv','importancia_clima.csv','importancia_clima_con_geografia.csv']  # Enumeramos tablas verificadas.
    tablas = [pd.read_csv(carpeta/n) for n in archivos]  # Leemos métricas reales desde CSV reproducibles.
    guardar_graficas(carpeta,*tablas)  # Exportamos las cinco gráficas con las mismas métricas.
    print('Gráficas guardadas en',carpeta)  # Informamos la carpeta de salida al equipo.
