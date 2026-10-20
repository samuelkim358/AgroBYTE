"""Punto de entrada de AgroBYTE: verificamos, procesamos, entrenamos y analizamos."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Habilitamos anotaciones modernas compatibles con nuestras herramientas.
import argparse  # Interpretamos las acciones solicitadas desde la terminal.
import subprocess  # Ejecutamos los módulos científicos con el mismo intérprete de Python.
import sys  # Localizamos el intérprete activo, incluyendo el entorno virtual del equipo.
from pathlib import Path  # Trabajamos con rutas independientes del sistema operativo.
from codigo.rutas import carpeta_dataset, raiz_proyecto, validar_carpeta_salida  # Compartimos las reglas de ubicación con todos los módulos.



# ==================== 2. CARPETAS COMPARTIDAS ====================
RAIZ = raiz_proyecto()  # Localizamos automáticamente AGROBYTE PROJECT en cualquier computadora.
SALIDA_FASE1 = RAIZ / 'salidas' / 'fase1'  # Concentramos los resultados de preparación y modelos iniciales.
SALIDA_FASE2 = RAIZ / 'salidas' / 'fase2'  # Concentramos los resultados de comparación avanzada.



# ==================== 3. EJECUCIÓN SEGURA DE UNA FASE ====================
def ejecutar(argumentos):  # Invocamos otro script sin construir comandos manuales dependientes del sistema.
    comando = [sys.executable, *[str(valor) for valor in argumentos]]  # Reutilizamos el Python donde instalamos nuestras dependencias.
    print('\nEjecutamos:', ' '.join(comando), flush=True)  # Mostramos el comando completo para facilitar el diagnóstico.
    subprocess.run(comando, cwd=RAIZ, check=True)  # Esperamos su finalización y propagamos cualquier error real.



# ==================== 4. MENÚ DE ACCIONES PARA EL EQUIPO ====================
def main():  # Permitimos trabajar con las mismas instrucciones desde cualquier computadora.
    cli = argparse.ArgumentParser(description='AgroBYTE: verificación, fase 1 y fase 2 reproducibles')  # Documentamos los comandos disponibles.
    cli.add_argument('accion', choices=['verificar', 'fase1', 'fase2', 'diagnostico', 'todo'])  # Seleccionamos la operación que vamos a ejecutar.
    cli.add_argument('--datos', type=Path, default=None, help='Ruta alternativa opcional del dataset')  # Conservamos una opción para pruebas externas.
    cli.add_argument('--reusar-resultados', action='store_true', help='Reutiliza solo parte de las comparaciones de Fase 2')  # Evitamos ejecutar de nuevo ciertos experimentos cuando así se solicita.
    opciones = cli.parse_args()  # Leemos los argumentos escritos en la terminal.
    datos = carpeta_dataset(opciones.datos)  # Buscamos por defecto DATASET_RETO_AGRO_2026 dentro del proyecto.
    salida1 = validar_carpeta_salida(datos, SALIDA_FASE1)  # Verificamos que la Fase 1 no escriba sobre archivos oficiales.
    salida2 = validar_carpeta_salida(datos, SALIDA_FASE2)  # Verificamos que la Fase 2 no escriba sobre archivos oficiales.
    codigo1 = RAIZ / 'codigo' / 'experimento_agrocebada.py'  # Localizamos el módulo de preparación y modelado inicial.
    codigo2 = RAIZ / 'codigo' / 'fase2_modelos_comparacion.py'  # Localizamos el módulo de comparación avanzada.
    diagnostico = RAIZ / 'codigo' / 'diagnosticos_fase2.py'  # Localizamos el módulo de evaluación adicional.
    print('Datos oficiales (solo lectura):', datos, flush=True)  # Confirmamos dónde se encuentra el dataset.
    print('Resultados dentro de:', RAIZ / 'salidas', flush=True)  # Confirmamos dónde aparecerán los nuevos archivos.
    if opciones.accion == 'verificar':  # Revisamos los archivos antes de ejecutar procesos costosos.
        ejecutar([codigo1, '--datos', datos, '--salida', salida1, '--solo-verificar'])  # Buscamos los insumos sin entrenar modelos.
    elif opciones.accion == 'fase1':  # Ejecutamos la etapa de ingeniería de características y validación inicial.
        ejecutar([codigo1, '--datos', datos, '--salida', salida1])  # Guardamos los resultados preliminares en su propia carpeta.
    elif opciones.accion == 'fase2':  # Ejecutamos la comparación y validación avanzada.
        if not (salida1 / 'caracteristicas_nubes_20.csv').exists():  # Exigimos la matriz generada por la fase anterior.
            raise FileNotFoundError('Primero ejecutamos: python ejecutar.py fase1')  # Explicamos cómo corregir el requisito faltante.
        argumentos = [codigo2, '--datos', datos, '--fase1', salida1, '--salida', salida2]  # Preparamos las rutas para la segunda etapa.
        if opciones.reusar_resultados:  # Respetamos la decisión opcional de reutilizar los cálculos guardados.
            argumentos.append('--reusar-resultados')  # Comunicamos la opción al módulo científico.
        ejecutar(argumentos)  # Entrenamos y contrastamos candidatos sin tocar los originales.
    elif opciones.accion == 'diagnostico':  # Revisamos resultados obtenidos después de ejecutar la Fase 2.
        ejecutar([diagnostico, '--datos', datos, '--salida', salida2])  # Comprobamos las predicciones y los diagnósticos por estado.
    else:  # Permitimos ejecutar las dos etapas completas en secuencia.
        ejecutar([codigo1, '--datos', datos, '--salida', salida1])  # Construimos primero las características.
        ejecutar([codigo2, '--datos', datos, '--fase1', salida1, '--salida', salida2])  # Comparamos modelos con los datos preparados.



# ==================== 5. INICIO DEL PROGRAMA ====================
if __name__ == '__main__':  # Evitamos lanzar experimentos cuando importamos este archivo desde pruebas.
    main()  # Iniciamos el menú de acuerdo con los parámetros de la terminal.
