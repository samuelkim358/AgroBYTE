"""Validamos y organizamos predicciones de AgroBYTE sin cargar modelos ni alterar originales."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Utilizamos anotaciones modernas en las funciones.
import hashlib  # Calculamos huellas para comprobar la identidad del CSV.
import json  # Leemos los manifiestos de trazabilidad del candidato.
from pathlib import Path  # Construimos rutas portables entre computadoras.
import numpy as np  # Validamos números y tolerancias de los cálculos.
import pandas as pd  # Leemos y analizamos las tablas del proyecto.


# ==================== 2. DEFINICIÓN DE RUTAS DEL PROYECTO ====================
RAIZ = Path(__file__).resolve().parent.parent  # Identificamos el directorio principal sin rutas personales.
ARCHIVO_PREDICCIONES = RAIZ / 'salidas' / 'fase4' / 'resultados' / 'predicciones_59_REVISION_EQUIPO.csv'  # Buscamos el CSV del modelo candidato.
ARCHIVO_MANIFIESTO = RAIZ / 'salidas' / 'fase4' / 'resultados' / 'manifiesto_candidato.json'  # Buscamos la evidencia de trazabilidad.
CARPETA_DATOS = RAIZ / 'DATASET_RETO_AGRO_2026'  # Buscamos los datos originales de FIRA, solamente para validación.
ARCHIVO_COORDENADAS = RAIZ / 'salidas' / 'fase1' / 'matriz_variables_197_parcelas.csv'  # Buscamos latitud y longitud para el mapa.
CARPETA_FASE3 = RAIZ / 'salidas' / 'fase3' / 'resultados'  # Buscamos la evidencia reproducida durante la Fase 3.
CARPETA_AUDITORIA = RAIZ / 'salidas' / 'auditoria_final'  # Buscamos los indicadores de cercanía geográfica opcionales.
COLUMNAS_REQUERIDAS = ('ID_POLIGONO', 'estado', 'municipio', 'AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'PRODUCCION_ESTIMADA_T')  # Fijamos el contrato de interfaz.


# ==================== 3. VALIDACIÓN DEL CSV OFICIAL DEL MODELO CANDIDATO ====================
def validar_predicciones(tabla: pd.DataFrame, ids_oficiales: set[str] | None = None) -> pd.DataFrame:  # Comprobamos cobertura y coherencia antes de mostrar predicciones.
    faltantes = set(COLUMNAS_REQUERIDAS).difference(tabla.columns)  # Buscamos columnas obligatorias ausentes.
    if faltantes:  # Detenemos la visualización cuando falta información indispensable.
        raise ValueError(f'Faltan columnas obligatorias: {sorted(faltantes)}')  # Indicamos exactamente qué falta.
    tabla = tabla.loc[:, list(COLUMNAS_REQUERIDAS)].copy()  # Conservamos únicamente columnas verificadas y no alteramos el original.
    tabla['ID_POLIGONO'] = tabla['ID_POLIGONO'].astype('string').str.strip()  # Normalizamos los identificadores para comparar fuentes.
    for campo in ('estado', 'municipio'):  # Revisamos las etiquetas de ubicación textual.
        tabla[campo] = tabla[campo].astype('string').str.strip()  # Evitamos diferencias por espacios al filtrar.
    for campo in ('AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'PRODUCCION_ESTIMADA_T'):  # Recorremos todas las variables numéricas.
        tabla[campo] = pd.to_numeric(tabla[campo], errors='coerce')  # Detectamos datos ilegibles como valores ausentes.
    if len(tabla) != 59:  # Exigimos el tamaño del conjunto oculto proporcionado por FIRA.
        raise ValueError(f'Se esperaban 59 parcelas de predicción y encontramos {len(tabla)}.')  # Evitamos mostrar entregas incompletas.
    if tabla['ID_POLIGONO'].isna().any() or tabla['ID_POLIGONO'].duplicated().any():  # Comprobamos identidad y unicidad de parcelas.
        raise ValueError('Existen identificadores de parcela vacíos o duplicados.')  # Rechazamos asociaciones ambiguas.
    if tabla[['estado', 'municipio']].isna().any().any() or (tabla[['estado', 'municipio']] == '').any().any():  # Evitamos regiones sin identificar.
        raise ValueError('Existen parcelas sin estado o municipio válido.')  # Explicamos el error de metadatos.
    valores = tabla[['AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'PRODUCCION_ESTIMADA_T']].to_numpy(dtype=float)  # Extraemos los números sin redondear.
    if not np.isfinite(valores).all() or not (valores > 0).all():  # Validamos números finitos y positivos.
        raise ValueError('Las superficies, los rendimientos y la producción deben ser finitos y positivos.')  # Evitamos resultados imposibles.
    produccion = tabla['AREA_HA'].to_numpy(dtype=float) * tabla['RENDIMIENTO_PREDICHO_T_HA'].to_numpy(dtype=float)  # Recalculamos toneladas totales.
    if not np.allclose(produccion, tabla['PRODUCCION_ESTIMADA_T'].to_numpy(dtype=float), rtol=0, atol=0.001):  # Verificamos toneladas = hectáreas por t/ha.
        raise ValueError('La producción total no coincide con AREA_HA × RENDIMIENTO_PREDICHO_T_HA.')  # Rechazamos archivos inconsistentes.
    if ids_oficiales is not None and set(tabla['ID_POLIGONO']) != ids_oficiales:  # Contrastamos todos los ID con el conjunto oculto oficial.
        raise ValueError('Los 59 identificadores no coinciden exactamente con los del dataset de FIRA.')  # Evitamos presentar predicciones de otras parcelas.
    return tabla.sort_values('ID_POLIGONO').reset_index(drop=True)  # Presentamos los resultados en un orden estable.


# ==================== 4. COMPROBACIÓN OPCIONAL DE IDENTIFICADORES OFICIALES ====================
def recuperar_ids_ocultos(carpeta: Path = CARPETA_DATOS) -> set[str] | None:  # Identificamos las parcelas del conjunto oculto sin consultar sus rendimientos.
    archivos = sorted(Path(carpeta).glob('ID_area_rendimiento*.csv'))  # Localizamos el archivo original usando su patrón conocido.
    if not archivos:  # Permitimos la visualización local incluso si todavía no se copió el dataset.
        return None  # Señalamos que la verificación contra la fuente oficial no está disponible.
    if len(archivos) != 1:  # Rechazamos fuentes ambiguas que podrían mezclar versiones.
        raise ValueError('Encontramos varios archivos de rendimientos oficiales; conservemos uno solo.')  # Solicitamos una fuente inequívoca.
    fuente = pd.read_csv(archivos[0], usecols=['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA'])  # Leemos identificadores y la separación oficial.
    ocultas = fuente.loc[fuente['CONJUNTO'].eq('PREDICCION')]  # Seleccionamos exclusivamente los casos sin etiqueta conocida.
    if len(ocultas) != 59 or not ocultas['RENDIMIENTO_T_HA'].isna().all():  # Comprobamos la separación 70/30 sin usar objetivos ocultos.
        raise ValueError('La tabla oficial no contiene exactamente 59 parcelas de PREDICCION sin rendimiento.')  # Evitamos cualquier fuga de etiquetas.
    return set(ocultas['ID_POLIGONO'].astype('string').str.strip())  # Devolvemos las identidades válidas para auditoría.


# ==================== 5. TRAZABILIDAD Y LECTURA DE PREDICCIONES ====================
def sha256(archivo: Path) -> str:  # Calculamos la huella de los bytes del archivo candidato.
    digestor = hashlib.sha256()  # Inicializamos el algoritmo SHA-256.
    with Path(archivo).open('rb') as entrada:  # Abrimos el CSV en modo lectura binaria.
        for bloque in iter(lambda: entrada.read(1024 * 1024), b''):  # Procesamos el CSV sin cargarlo entero en memoria.
            digestor.update(bloque)  # Incorporamos el contenido de cada bloque a la firma.
    return digestor.hexdigest()  # Devolvemos la firma para contrastarla con el manifiesto.


def cargar_predicciones(archivo: Path = ARCHIVO_PREDICCIONES, carpeta_oficial: Path = CARPETA_DATOS) -> tuple[pd.DataFrame, dict]:  # Reunimos datos, controles y advertencias sin entrenar.
    archivo = Path(archivo)  # Convertimos la entrada a una ruta común.
    if not archivo.is_file():  # Detectamos cuando la Fase 4 no se ha ejecutado todavía.
        raise FileNotFoundError(f'No encontramos {archivo}. Ejecutamos primero: python fase4.py')  # Indicamos el comando necesario.
    ids = recuperar_ids_ocultos(carpeta_oficial)  # Validamos el conjunto oficial cuando está disponible.
    predicciones = validar_predicciones(pd.read_csv(archivo), ids)  # Leemos el CSV y verificamos sus resultados.
    manifiesto_path = archivo.parent / 'manifiesto_candidato.json'  # Localizamos el manifiesto junto al CSV elegido.
    manifiesto = json.loads(manifiesto_path.read_text(encoding='utf-8')) if manifiesto_path.is_file() else {}  # Leemos la trazabilidad sin cargar un joblib.
    advertencias = []  # Registramos limitaciones visibles para el usuario de la interfaz.
    if ids is None:  # Detectamos falta de comprobación contra la fuente oficial.
        advertencias.append('No encontramos el dataset oficial para contrastar identificadores; las otras verificaciones sí se aplicaron.')  # Informamos sin falsear la validación.
    if not manifiesto:  # Detectamos ausencia de la firma del entrenamiento.
        advertencias.append('No encontramos manifiesto de la Fase 4; la procedencia no está completamente verificada.')  # Diferenciamos integridad de trazabilidad.
    elif manifiesto.get('sha256_csv_completo') != sha256(archivo):  # Contrastamos la versión exacta del archivo.
        advertencias.append('La huella SHA-256 del CSV no coincide con el manifiesto; revisamos versiones antes de compartirlo.')  # Evitamos afirmar procedencia incorrecta.
    return predicciones, {'manifiesto': manifiesto, 'advertencias': advertencias, 'fuente_oficial_comprobada': ids is not None, 'hash_verificado': bool(manifiesto) and manifiesto.get('sha256_csv_completo') == sha256(archivo)}  # Entregamos datos junto a sus evidencias.


# ==================== 6. DATOS DE CONTEXTO PARA MAPA Y AUDITORÍA ====================
def unir_coordenadas(tabla: pd.DataFrame, archivo: Path = ARCHIVO_COORDENADAS) -> pd.DataFrame:  # Asociamos centroides sin calcular rendimiento de nuevo.
    tabla = tabla.copy()  # Evitamos cambios involuntarios en los datos ya comprobados.
    if not Path(archivo).is_file():  # Permitimos utilizar el dashboard aunque falte la matriz de la Fase 1.
        return tabla  # Mostramos los rendimientos y omitimos solamente el mapa.
    geo = pd.read_csv(archivo, usecols=['ID_POLIGONO', 'latitud', 'longitud'])  # Recuperamos las coordenadas derivadas previamente de los polígonos.
    if geo['ID_POLIGONO'].duplicated().any():  # Verificamos que cada punto geográfico sea único.
        raise ValueError('La matriz de la Fase 1 repite identificadores geográficos.')  # Rechazamos emparejamientos incorrectos.
    tabla = tabla.merge(geo, on='ID_POLIGONO', how='left', validate='one_to_one')  # Unimos los puntos exclusivamente por identificador.
    if not np.isfinite(tabla[['latitud', 'longitud']].to_numpy(float)).all():  # Buscamos centroides faltantes o inválidos.
        raise ValueError('Existen parcelas ocultas sin coordenadas geográficas válidas.')  # Evitamos ubicar puntos incorrectos.
    return tabla  # Entregamos los valores de la predicción junto a su ubicación.


def cargar_auditoria_geografica(archivo: Path = CARPETA_AUDITORIA / 'auditoria_59_parcelas.csv') -> pd.DataFrame | None:  # Recuperamos distancias y banderas si se realizó la auditoría.
    if not Path(archivo).is_file():  # Permitimos que la interfaz funcione sin la auditoría opcional.
        return None  # Indicamos que estos indicadores todavía no se han cargado.
    return pd.read_csv(archivo)  # Leemos únicamente el resumen ya calculado sin modificar fuentes.


def cargar_validacion(carpeta: Path = CARPETA_FASE3) -> dict[str, pd.DataFrame | None]:  # Buscamos métricas derivadas de la Fase 3.
    archivos = {'ranking': 'ranking_exploratorio_fase3.csv', 'espacial': 'comparacion_municipios_fase3.csv', 'anidada': 'evaluacion_anidada_fase3.csv', 'selecciones': 'selecciones_internas_fase3.csv'}  # Definimos las fuentes verificables de las métricas.
    return {nombre: pd.read_csv(carpeta / archivo) if (carpeta / archivo).is_file() else None for nombre, archivo in archivos.items()}  # Cargamos resultados existentes sin inventar valores.


# ==================== 7. RESÚMENES Y EXPORTACIONES ====================
def resumen_indicadores(tabla: pd.DataFrame) -> dict:  # Calculamos indicadores únicamente para el subconjunto filtrado.
    area = float(tabla['AREA_HA'].sum())  # Sumamos la superficie en hectáreas.
    produccion = float(tabla['PRODUCCION_ESTIMADA_T'].sum())  # Sumamos las toneladas estimadas.
    return {'parcelas': int(len(tabla)), 'area_ha': area, 'produccion_t': produccion, 'rendimiento_ponderado_t_ha': produccion / area if area > 0 else float('nan')}  # Calculamos el promedio ponderado y evitamos dividir entre cero.


def csv_exportacion(tabla: pd.DataFrame, minimo: bool = False) -> bytes:  # Construimos un archivo para descargar, sin alterar el CSV candidato.
    campos = ['ID_POLIGONO', 'RENDIMIENTO_PREDICHO_T_HA'] if minimo else list(COLUMNAS_REQUERIDAS)  # Elegimos una exportación mínima o detallada.
    return tabla.loc[:, campos].to_csv(index=False, float_format='%.9f').encode('utf-8-sig')  # Codificamos UTF-8 compatible con Excel en Windows.
