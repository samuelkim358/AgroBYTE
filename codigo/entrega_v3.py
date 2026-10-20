"""Cargamos el artefacto aprobado y auditamos cada inferencia de AgroBYTE 3.0."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Habilitamos anotaciones modernas para nuestras funciones.
import hashlib  # Calculamos firmas criptográficas de los recursos congelados.
import json  # Leemos metadatos y polígonos en un formato portable.
import warnings  # Registramos advertencias al cargar un modelo de otra versión.
from pathlib import Path  # Evitamos rutas absolutas específicas de una computadora.
import joblib  # Cargamos el pipeline original de Extra Trees sin reentrenarlo.
import numpy as np  # Evaluamos tolerancias numéricas e integridad de predicciones.
import pandas as pd  # Relacionamos parcelas, coordenadas y rendimientos.


# ==================== 2. ARCHIVOS INMUTABLES DE LA ENTREGA ====================
RAIZ = Path(__file__).resolve().parent.parent  # Identificamos la carpeta principal a partir del archivo.
DATOS = RAIZ / 'datos'  # Concentramos los insumos mínimos de las 59 parcelas.
MODELO = RAIZ / 'modelo' / 'modelo_extratrees_geografico.joblib'  # Usamos nuestro único modelo seleccionado.
CAMPOS = ['AREA_HA', 'latitud', 'longitud', 'estado', 'municipio']  # Conservamos el orden exacto de entrenamiento.
COLUMNAS = ['ID_POLIGONO', 'estado', 'municipio', 'AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'PRODUCCION_ESTIMADA_T']  # Definimos la salida auditada.


# ==================== 3. COMPROBACIÓN DE HUELLAS SHA-256 ====================
def sha256(ruta: Path) -> str:  # Obtenemos una firma estable para detectar cambios accidentales.
    hash_obj = hashlib.sha256()  # Inicializamos el cálculo criptográfico.
    with ruta.open('rb') as archivo:  # Leemos el recurso binario sin modificarlo.
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):  # Dividimos los bytes para ahorrar memoria.
            hash_obj.update(bloque)  # Incorporamos cada segmento del archivo.
    return hash_obj.hexdigest()  # Entregamos la firma hexadecimal final.


def verificar_huellas(manifiesto: dict) -> None:  # Contrastamos todo archivo requerido con el manifiesto.
    for nombre, esperado in manifiesto['sha256'].items():  # Recorremos cada recurso de inferencia.
        ruta = RAIZ / nombre  # Construimos su ruta dentro del paquete distribuido.
        if not ruta.exists():  # Detectamos archivos faltantes antes de ejecutar cálculos.
            raise FileNotFoundError(f'Falta el archivo requerido: {nombre}')  # Detenemos la aplicación si está incompleta.
        if sha256(ruta) != esperado:  # Detectamos cambios respecto a nuestra versión congelada.
            raise ValueError(f'La firma SHA-256 no coincide para {nombre}')  # Evitamos publicar números que no auditamos.


# ==================== 4. VALIDACIÓN DEL CSV Y COORDENADAS ====================
def validar_tablas(csv: pd.DataFrame, entradas: pd.DataFrame) -> pd.DataFrame:  # Exigimos filas unívocas y consistencia física.
    if set(COLUMNAS) - set(csv.columns):  # Comprobamos la existencia de los campos de la entrega.
        raise ValueError('El CSV carece de columnas de predicción requeridas.')  # Evitamos inferir columnas desconocidas.
    if set(['ID_POLIGONO', *CAMPOS]) - set(entradas.columns):  # Comprobamos la estructura de inferencia.
        raise ValueError('Faltan columnas de entrada necesarias para Extra Trees.')  # Impedimos ejecutar características incompletas.
    if len(csv) != 59 or len(entradas) != 59:  # Exigimos el número oficial de parcelas ocultas.
        raise ValueError('Se requieren exactamente 59 parcelas.')  # Rechazamos exportaciones parciales.
    for tabla in (csv, entradas):  # Recorremos ambos archivos relacionados.
        if tabla.ID_POLIGONO.isna().any() or tabla.ID_POLIGONO.duplicated().any():  # Comprobamos claves no vacías y únicas.
            raise ValueError('Hay IDs vacíos o repetidos.')  # Evitamos duplicaciones de predicciones.
    if set(csv.ID_POLIGONO) != set(entradas.ID_POLIGONO):  # Aseguramos la cobertura por identificador.
        raise ValueError('Los identificadores del CSV no coinciden con las entradas.')  # Bloqueamos asociaciones incorrectas.
    tabla = entradas.merge(csv[COLUMNAS], on='ID_POLIGONO', suffixes=('_entrada',''), validate='one_to_one')  # Relacionamos datos sin emparejar por posición.
    for campo in ('estado', 'municipio'):  # Comprobamos que ubicación no cambió entre fuentes.
        if not (tabla[f'{campo}_entrada'].astype(str) == tabla[campo].astype(str)).all():  # Detectamos lugares incorrectos.
            raise ValueError(f'Las etiquetas de {campo} son inconsistentes.')  # Conservamos identidad regional.
    if not np.allclose(tabla['AREA_HA_entrada'], tabla['AREA_HA'], atol=1e-7, rtol=0):  # Comprobamos que superficie no cambió.
        raise ValueError('La superficie del CSV difiere de la usada por el modelo.')  # Evitamos errores en toneladas totales.
    valores = tabla[['AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T','latitud','longitud']].to_numpy(dtype=float)  # Convertimos campos para evaluar su dominio.
    if not np.isfinite(valores).all() or not (tabla['AREA_HA'] > 0).all() or not (tabla['RENDIMIENTO_PREDICHO_T_HA'] > 0).all():  # Validamos positividad y finitud.
        raise ValueError('Hay superficies, coordenadas o rendimientos inválidos.')  # Rechazamos entradas ilegibles.
    estimada = tabla['AREA_HA'] * tabla['RENDIMIENTO_PREDICHO_T_HA']  # Aplicamos superficie por rendimiento.
    if not np.allclose(estimada, tabla['PRODUCCION_ESTIMADA_T'], atol=1e-6, rtol=0):  # Comprobamos precisión de exportación.
        raise ValueError('Las toneladas totales no coinciden con área por rendimiento.')  # Evitamos errores de unidades.
    return tabla.sort_values('ID_POLIGONO').reset_index(drop=True)  # Estabilizamos el orden de visualización.


# ==================== 5. INFERENCIA REAL Y REPRODUCCIÓN DEL CANDIDATO ====================
def cargar_entrega() -> tuple[pd.DataFrame, dict, object, list[str]]:  # Entregamos datos auditados y modelo listo para consultas.
    manifiesto = json.loads((DATOS / 'version_entrega.json').read_text(encoding='utf-8'))  # Recuperamos nuestra versión técnica.
    verificar_huellas(manifiesto)  # Confirmamos que el modelo y los datos no cambiaron.
    csv = pd.read_csv(DATOS / 'predicciones_59.csv')  # Leemos estimaciones congeladas del 2.0.
    entradas = pd.read_csv(DATOS / 'entradas_59.csv')  # Leemos las cinco características de las 59 parcelas.
    tabla = validar_tablas(csv, entradas)  # Validamos la relación de superficies y rendimientos.
    with warnings.catch_warnings(record=True) as avisos:  # Registramos posibles incompatibilidades entre versiones.
        warnings.simplefilter('always')  # Capturamos todos los avisos de carga relevantes.
        modelo = joblib.load(MODELO)  # Cargamos exclusivamente nuestro archivo local, nunca uno enviado por usuarios.
    columnas = list(manifiesto['caracteristicas'])  # Recuperamos el contrato de variables del entrenamiento.
    if columnas != CAMPOS:  # Confirmamos el orden de entrada establecido científicamente.
        raise ValueError('Las columnas del manifiesto no coinciden con las características auditadas.')  # Bloqueamos entrada reordenada.
    regeneradas = np.asarray(modelo.predict(tabla[CAMPOS]), dtype=float)  # Ejecutamos de nuevo el pipeline real para las 59 parcelas.
    esperadas = tabla['RENDIMIENTO_PREDICHO_T_HA'].to_numpy(dtype=float)  # Recuperamos predicciones firmadas del equipo.
    if not np.allclose(regeneradas, esperadas, rtol=0, atol=1e-6):  # Comprobamos que el modelo serializado conserva sus resultados.
        raise ValueError('El modelo activo NO reproduce las 59 predicciones auditadas. Revisamos bibliotecas y serialización.')  # Evitamos desplegar inferencias inconsistentes.
    tabla['RENDIMIENTO_EN_VIVO_T_HA'] = regeneradas  # Asociamos las inferencias actuales a la tabla auditada.
    etiquetas = sorted({str(x.message) for x in avisos if 'version' in str(x.message).lower()})  # Conservamos mensajes de incompatibilidad de versión.
    return tabla, manifiesto, modelo, etiquetas  # Permitimos consultas y descargas con evidencia.


def predecir_parcela(modelo: object, tabla: pd.DataFrame, id_parcela: str) -> tuple[float, float]:  # Ejecutamos inferencia para una parcela oficial elegida.
    subset = tabla.loc[tabla['ID_POLIGONO'].eq(id_parcela)]  # Buscamos por clave, no por índice accidental.
    if len(subset) != 1:  # Restringimos la función a una parcela oficial reconocida.
        raise ValueError('Seleccionamos una parcela oficial válida de las 59.')  # Rechazamos datos arbitrarios no validados.
    pronostico = float(modelo.predict(subset[CAMPOS])[0])  # Ejecutamos el pipeline cargado con cinco variables originales.
    return pronostico, pronostico * float(subset['AREA_HA'].iloc[0])  # Devolvemos toneladas por hectárea y toneladas totales.


# ==================== 6. SIMULACIÓN ECONÓMICA SIN SUPUESTOS OCULTOS ====================
def simular(area: float, rinde: float, precio_t: float, semilla_ha: float, fertilizante_ha: float, labor_ha: float, otro_ha: float, credito: float = 0, tasa: float = 0, meses: int = 0) -> dict:  # Calculamos un escenario financiero transparente.
    valores = [area, rinde, precio_t, semilla_ha, fertilizante_ha, labor_ha, otro_ha, credito, tasa]  # Reunimos las entradas de negocio.
    if not all(np.isfinite(v) and v >= 0 for v in valores) or meses < 0:  # Rechazamos costos y tiempos inválidos.
        raise ValueError('Los montos, la superficie y los períodos deben ser no negativos y finitos.')  # Protegemos cálculos económicos.
    produccion = area * rinde  # Calculamos el volumen proyectado en toneladas.
    ingreso = produccion * precio_t  # Calculamos ventas brutas bajo el precio supuesto.
    costos = area * (semilla_ha + fertilizante_ha + labor_ha + otro_ha)  # Sumamos únicamente los rubros ingresados.
    interes_simple = credito * (tasa / 100) * meses / 12  # Ilustramos interés simple, sin simular condiciones reales de un crédito.
    margen = ingreso - costos - interes_simple  # Calculamos excedente antes de impuestos y costos omitidos.
    costo_equilibrio = (costos + interes_simple) / (area * precio_t) if area > 0 and precio_t > 0 else None  # Obtenemos rendimiento de equilibrio.
    return dict(produccion_t=produccion, ingreso_mxn=ingreso, costo_mxn=costos, interes_simple_mxn=interes_simple, margen_estimado_mxn=margen, rendimiento_equilibrio_t_ha=costo_equilibrio)  # Entregamos indicadores que no son aprobación crediticia.
