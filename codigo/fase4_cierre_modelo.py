"""Preparamos el modelo candidato de AgroBYTE sin modificar las Fases 1, 2 y 3."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Utilizamos anotaciones modernas sin exigir evaluarlas al importar el módulo.
import argparse  # Interpretamos las opciones que escribe el equipo en la terminal.
import hashlib  # Calculamos huellas digitales para verificar los archivos de entrada y salida.
import json  # Guardamos la evidencia del entrenamiento en un manifiesto legible.
import platform  # Registramos la versión de Python con la que ejecutamos el modelo.
import sys  # Localizamos los módulos del proyecto sin incluir rutas personales.
from datetime import datetime, timezone  # Registramos la fecha y hora UTC del entrenamiento.
from pathlib import Path  # Construimos rutas que funcionan en Windows, macOS y Linux.
import joblib  # Conservamos juntos el preprocesamiento y el modelo entrenado.
import numpy as np  # Comprobamos que las predicciones contienen números válidos.
import pandas as pd  # Organizamos las 59 predicciones en archivos CSV.
import sklearn  # Registramos la versión utilizada del sistema de Machine Learning.



# ==================== 2. IMPORTACIÓN DE CÓDIGO YA VALIDADO ====================
RAIZ = Path(__file__).resolve().parent.parent  # Identificamos el directorio del proyecto a partir de este archivo.
sys.path.insert(0, str(RAIZ / 'codigo'))  # Importamos los módulos previos desde una carpeta conocida.
import fase3_xgboost as fase3  # Reutilizamos sus controles de integridad y selección de variables.
from rutas import carpeta_dataset, validar_carpeta_salida  # Evitamos escribir en la carpeta original de FIRA.



# ==================== 3. CONFIGURACIÓN DEL CANDIDATO ====================
NOMBRE_CANDIDATO = 'geo_extratrees'  # Conservamos el candidato elegido en los cinco folds de la evaluación anidada.
NOMBRE_MODELO = 'extratrees'  # Utilizamos la misma implementación de árboles que en las fases previas.
GRUPO_VARIABLES = 'geografia'  # Usamos las cinco características del experimento mejor evaluado por parcelas.
ARCHIVO_MATRIZ = 'caracteristicas_nubes_20.csv'  # Fijamos el archivo de características empleado durante la comparación.



# ==================== 4. UTILIDADES DE INTEGRIDAD ====================
def huella_sha256(archivo: Path) -> str:  # Creamos una huella para distinguir el contenido exacto de cada archivo.
    digestor = hashlib.sha256()  # Inicializamos el algoritmo de integridad SHA-256.
    with Path(archivo).open('rb') as origen:  # Leemos el archivo en modo binario, sin modificarlo.
        for bloque in iter(lambda: origen.read(1024 * 1024), b''):  # Procesamos bloques pequeños para ahorrar memoria.
            digestor.update(bloque)  # Incorporamos los bytes de cada bloque a la huella.
    return digestor.hexdigest()  # Devolvemos la huella con la que podremos comparar ejecuciones.



def comprobar_predicciones(tabla: pd.DataFrame, identificadores: set[str]) -> None:  # Validamos cada predicción antes de exportarla.
    if len(tabla) != 59 or tabla['ID_POLIGONO'].nunique() != 59:  # Exigimos exactamente 59 parcelas sin duplicados.
        raise ValueError('La tabla debe contener 59 identificadores únicos.')  # Evitamos un archivo incompleto o repetido.
    if set(tabla['ID_POLIGONO']) != identificadores:  # Contrastamos los ID contra el conjunto oficial sin rendimiento.
        raise ValueError('Los identificadores no coinciden con las parcelas ocultas de FIRA.')  # Rechazamos asociaciones erróneas.
    if not (tabla['AREA_HA'].gt(0).all() and np.isfinite(tabla['AREA_HA'].to_numpy(float)).all()):  # Exigimos superficies reales positivas.
        raise ValueError('Se encontraron superficies inválidas.')  # Detenemos cálculos físicamente imposibles.
    valores = tabla['RENDIMIENTO_PREDICHO_T_HA'].to_numpy(float)  # Extraemos los rendimientos en toneladas por hectárea.
    if not (np.isfinite(valores).all() and (valores > 0).all()):  # Impedimos predicciones no numéricas o negativas.
        raise ValueError('Se encontraron rendimientos predichos inválidos.')  # Evitamos exportar un archivo defectuoso.
    esperada = tabla['AREA_HA'].to_numpy(float) * valores  # Recalculamos la producción total a partir de las hectáreas.
    if not np.allclose(tabla['PRODUCCION_ESTIMADA_T'].to_numpy(float), esperada, atol=1e-9):  # Verificamos las unidades y la multiplicación.
        raise ValueError('La producción total no coincide con superficie por rendimiento.')  # Señalamos errores en la calculadora.



# ==================== 5. ENTRENAMIENTO DEL CANDIDATO CON LAS 138 ETIQUETAS ====================
def construir_resultado(datos: Path, matriz_fase1: Path):  # Recuperamos datos y obtenemos predicciones con nuestro pipeline validado.
    entrenamiento, ocultas = fase3.leer_parcelas(datos, matriz_fase1)  # Utilizamos las mismas fuentes y verificaciones de la Fase 3.
    columnas = fase3.fase2.columnas_para(entrenamiento, GRUPO_VARIABLES)  # Recuperamos exactamente el grupo geográfico utilizado.
    X_entrenamiento = entrenamiento[columnas]  # Preparamos exclusivamente las características de los 138 ejemplos conocidos.
    y_entrenamiento = entrenamiento['RENDIMIENTO_T_HA'].to_numpy(float)  # Separamos únicamente sus rendimientos observados.
    X_ocultas = ocultas[columnas]  # Preparamos las mismas cinco variables para las 59 parcelas por estimar.
    modelo = fase3.construir_candidato(NOMBRE_MODELO, X_entrenamiento)  # Reutilizamos los hiperparámetros y transformaciones del experimento.
    modelo.fit(X_entrenamiento, y_entrenamiento)  # Entrenamos con todas las parcelas etiquetadas y nunca con las 59 ocultas.
    estimaciones = np.asarray(modelo.predict(X_ocultas), dtype=float)  # Calculamos rendimiento en toneladas por hectárea.
    tabla = ocultas[['ID_POLIGONO', 'estado', 'municipio', 'AREA_HA']].copy()  # Conservamos identificadores y metadatos útiles para el dashboard.
    tabla['RENDIMIENTO_PREDICHO_T_HA'] = estimaciones  # Registramos la salida exacta del modelo sin redondeo prematuro.
    tabla['PRODUCCION_ESTIMADA_T'] = tabla['AREA_HA'].to_numpy(float) * estimaciones  # Calculamos toneladas esperadas por parcela.
    tabla = tabla.sort_values('ID_POLIGONO').reset_index(drop=True)  # Estabilizamos el orden para facilitar auditoría y comparación.
    comprobar_predicciones(tabla, set(ocultas['ID_POLIGONO']))  # Comprobamos cobertura, unidades e integridad de las estimaciones.
    return modelo, X_ocultas, tabla, columnas  # Devolvemos el modelo y la evidencia necesaria para verificarlo.



# ==================== 6. EXPORTACIÓN Y TRAZABILIDAD ====================
def exportar(datos: Path, fase1: Path, salida: Path):  # Generamos un paquete candidato separado de los borradores antiguos.
    destino = validar_carpeta_salida(datos, salida)  # Impedimos escribir en los documentos originales de FIRA.
    archivos = destino / 'resultados'  # Separamos tablas y trazabilidad del modelo serializado.
    modelos = destino / 'modelos'  # Reservamos una carpeta exclusiva para el modelo aprendido.
    if any((destino / ruta).exists() for ruta in ['resultados/predicciones_59_REVISION_EQUIPO.csv', 'modelos/modelo_candidato_extratrees.joblib']):  # Protegemos ejecuciones anteriores.
        raise FileExistsError('Ya hay un candidato: guardamos una copia o cambiamos --salida antes de repetir.')  # Evitamos sobrescrituras accidentales.
    archivos.mkdir(parents=True, exist_ok=True)  # Creamos el directorio de resultados si no existe.
    modelos.mkdir(parents=True, exist_ok=True)  # Creamos el directorio de modelos si no existe.
    modelo, X_ocultas, tabla, columnas = construir_resultado(datos, fase1)  # Reentrenamos y calculamos las 59 predicciones verificadas.
    modelo_path = modelos / 'modelo_candidato_extratrees.joblib'  # Elegimos un nombre que identifica el estado provisional.
    joblib.dump(modelo, modelo_path)  # Guardamos el pipeline con imputación, codificación y regresor aprendidos.
    reconstruido = joblib.load(modelo_path)  # Reabrimos exclusivamente nuestro propio archivo para comprobarlo.
    reestimaciones = reconstruido.predict(X_ocultas)  # Comprobamos que el modelo guardado produce idénticas predicciones.
    if not np.allclose(reestimaciones, tabla['RENDIMIENTO_PREDICHO_T_HA'].to_numpy(float), rtol=0, atol=1e-12):  # Exigimos concordancia estricta.
        raise ValueError('El modelo serializado no reproduce sus predicciones originales.')  # Impedimos entregar un modelo inconsistente.
    completa = archivos / 'predicciones_59_REVISION_EQUIPO.csv'  # Preparamos una tabla detallada para la aplicación web.
    minima = archivos / 'predicciones_59_MINIMO_REVISION_EQUIPO.csv'  # Preparamos una tabla reducida para posible entrega.
    tabla.to_csv(completa, index=False, float_format='%.9f')  # Exportamos información geográfica, rendimiento y producción total.
    tabla[['ID_POLIGONO', 'RENDIMIENTO_PREDICHO_T_HA']].to_csv(minima, index=False, float_format='%.9f')  # Exportamos un identificador y una predicción por fila.
    objetivo_encontrado = sorted(Path(datos).glob('ID_area_rendimiento*.csv'))  # Identificamos el archivo objetivo usado en el entrenamiento.
    if len(objetivo_encontrado) != 1:  # Confirmamos que la procedencia de etiquetas es inequívoca.
        raise FileNotFoundError('No se encontró un único CSV de rendimientos.')  # Protegemos el manifiesto de fuentes ambiguas.
    manifiesto = {  # Reunimos las condiciones suficientes para rastrear la versión entrenada.
        'estado': 'CANDIDATO_PENDIENTE_REVISION_EQUIPO_NO_ENTREGA',  # Diferenciamos el candidato del archivo formal publicado a FIRA.
        'generado_utc': datetime.now(timezone.utc).isoformat(),  # Registramos el momento de entrenamiento.
        'algoritmo': NOMBRE_MODELO,  # Indicamos la familia algorítmica utilizada.
        'candidato': NOMBRE_CANDIDATO,  # Identificamos la configuración ganadora de las validaciones por parcelas.
        'grupo_variables': GRUPO_VARIABLES,  # Registramos el criterio de selección de características.
        'columnas': columnas,  # Enumeramos explícitamente las entradas esperadas por el pipeline.
        'etiquetas_entrenamiento': 138,  # Documentamos cuántos rendimientos reales utilizamos.
        'parcelas_prediccion': 59,  # Documentamos cuántos rendimientos ocultos se estimaron.
        'metricas_validacion_anidada_fase3': {'rmse_t_ha': 0.5033839525043555, 'mae_t_ha': 0.35149652378708246, 'r2': 0.6476826446670407},  # Citamos la validación de la Fase 3, NO la evaluación oficial.
        'versiones': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},  # Registramos las bibliotecas activas.
        'sha256_dataset_objetivo': huella_sha256(objetivo_encontrado[0]),  # Conservamos la huella de la tabla de rendimiento original.
        'sha256_matriz_fase1': huella_sha256(Path(fase1) / ARCHIVO_MATRIZ),  # Conservamos la huella de la matriz transformada.
        'sha256_modelo': huella_sha256(modelo_path),  # Identificamos el archivo de modelo generado.
        'sha256_csv_completo': huella_sha256(completa),  # Comprobamos la identidad del archivo para el dashboard.
        'sha256_csv_minimo': huella_sha256(minima),  # Comprobamos la identidad del formato mínimo.
        'advertencias': ['Las métricas corresponden a validación con 138 etiquetas; no son de FIRA.', 'El modelo geográfico interpola entre zonas representadas; generaliza peor a municipios desconocidos.', 'No está validado para predicción previa a la siembra ni para parcelas arbitrarias.', 'Las predicciones deben ser revisadas por el equipo antes de entregarlas.'],  # Dejamos claras las limitaciones conocidas.
    }  # Terminamos el manifiesto de integridad del experimento.
    (archivos / 'manifiesto_candidato.json').write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding='utf-8')  # Guardamos un registro humano y verificable.
    print('CANDIDATO:', NOMBRE_CANDIDATO, flush=True)  # Identificamos el modelo seleccionado durante la ejecución.
    print('PARCELAS: 138 entrenadas; 59 predichas', flush=True)  # Confirmamos la separación de los dos conjuntos oficiales.
    print('RANGO ESTIMADO:', float(tabla['RENDIMIENTO_PREDICHO_T_HA'].min()), float(tabla['RENDIMIENTO_PREDICHO_T_HA'].max()), flush=True)  # Informamos únicamente el rango de predicciones.
    print('RESULTADOS:', archivos, flush=True)  # Mostramos dónde debe consultar los archivos el equipo.
    print('AVISO: pendiente de revisión del equipo; NO subir todavía a FIRA', flush=True)  # Evitamos confundir el candidato con la entrega aprobada.
    return tabla  # Permitimos a las pruebas automatizadas inspeccionar el archivo resultante.



# ==================== 7. PUNTO DE ENTRADA DE LA TERMINAL ====================
def main():  # Interpretamos rutas sin tener que editar el código en cada computadora.
    cli = argparse.ArgumentParser(description='AgroBYTE: cerramos el modelo candidato sin modificar las fases anteriores')  # Describimos la finalidad científica del comando.
    cli.add_argument('--datos', type=Path, default=None)  # Permitimos utilizar una copia alternativa del dataset.
    cli.add_argument('--fase1', type=Path, default=RAIZ / 'salidas' / 'fase1')  # Utilizamos la matriz de características ya generada.
    cli.add_argument('--salida', type=Path, default=RAIZ / 'salidas' / 'fase4')  # Guardamos todo dentro de una fase independiente.
    argumentos = cli.parse_args()  # Leemos los parámetros seleccionados por el equipo.
    datos = carpeta_dataset(argumentos.datos)  # Localizamos el conjunto oficial de solo lectura.
    exportar(datos, argumentos.fase1, argumentos.salida)  # Construimos las predicciones candidatas y su archivo de trazabilidad.


if __name__ == '__main__':  # Evitamos entrenar el modelo cuando otro módulo importa este archivo.
    main()  # Ejecutamos el entrenamiento candidato solo cuando así se solicita.
