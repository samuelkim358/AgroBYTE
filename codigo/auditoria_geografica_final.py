"""Auditamos la cobertura espacial de las 59 parcelas FIRA sin reentrenar el modelo."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Habilitamos anotaciones para facilitar la lectura del código.
import argparse  # Leemos las rutas que se proporcionan desde la terminal.
import json  # Guardamos los indicadores de auditoría en un formato estructurado.
from pathlib import Path  # Trabajamos con directorios independientes del sistema operativo.
import geopandas as gpd  # Abrimos las geometrías oficiales de las parcelas.
import matplotlib  # Preparamos gráficos que funcionan sin una ventana gráfica.
matplotlib.use('Agg')  # Permitimos generar figuras desde una terminal sin pantalla.
import matplotlib.pyplot as plt  # Dibujamos la cobertura espacial y los resultados.
import numpy as np  # Calculamos distancias y estadísticas numéricas.
import pandas as pd  # Combinamos parcelas, rendimientos y predicciones.
from shapely.geometry import MultiPoint, Point  # Comprobamos si las parcelas están dentro de las envolventes de referencia.



# ==================== 2. CONFIGURACIÓN Y RUTAS ====================
RAIZ = Path(__file__).resolve().parent.parent  # Localizamos la raíz del proyecto a partir de la ubicación del módulo.
DATOS_PREDETERMINADOS = RAIZ / 'DATASET_RETO_AGRO_2026'  # Buscamos los insumos originales en la carpeta común del equipo.
PREDICCIONES_PREDETERMINADAS = RAIZ / 'salidas' / 'fase4' / 'resultados' / 'predicciones_59_REVISION_EQUIPO.csv'  # Recuperamos el borrador de la Fase 4.
SALIDA_PREDETERMINADA = RAIZ / 'salidas' / 'auditoria_final'  # Guardamos nuestras comprobaciones en una carpeta independiente.
RADIO_TERRESTRE_KM = 6371.0088  # Aproximamos las distancias geográficas utilizando el radio medio terrestre.



# ==================== 3. DISTANCIAS SOBRE LA SUPERFICIE TERRESTRE ====================
def distancias_haversine_km(origen: np.ndarray, destino: np.ndarray) -> np.ndarray:  # Calculamos distancias entre dos conjuntos de centroides expresados en grados.
    a = np.radians(np.asarray(origen, dtype=float))  # Transformamos los grados [latitud, longitud] del origen a radianes.
    b = np.radians(np.asarray(destino, dtype=float))  # Transformamos los grados [latitud, longitud] del destino a radianes.
    dlat = b[None, :, 0] - a[:, None, 0]  # Calculamos las diferencias de latitud entre todas las parejas.
    dlon = b[None, :, 1] - a[:, None, 1]  # Calculamos las diferencias de longitud entre todas las parejas.
    valor = np.sin(dlat / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2) ** 2  # Aplicamos la fórmula de Haversine.
    angulo = 2 * np.arcsin(np.sqrt(np.clip(valor, 0, 1)))  # Calculamos el ángulo central y controlamos errores de redondeo.
    return RADIO_TERRESTRE_KM * angulo  # Convertimos el ángulo central en kilómetros aproximados.



# ==================== 4. CARGA Y VERIFICACIÓN DE LOS ARCHIVOS FIRA ====================
def cargar_parcelas(carpeta_datos: Path) -> pd.DataFrame:  # Reunimos geometrías y etiquetas de entrenamiento mediante identificadores estables.
    archivos = sorted(carpeta_datos.glob('ID_area_rendimiento*.csv'))  # Buscamos el único archivo oficial con rendimientos.
    if len(archivos) != 1:  # Evitamos leer un archivo incorrecto o ambiguo.
        raise FileNotFoundError(f'Esperamos un CSV de rendimientos en {carpeta_datos}; encontramos {len(archivos)}.')  # Informamos cómo corregir los insumos.
    objetivo = pd.read_csv(archivos[0])  # Leemos las 197 filas originales sin escribir nada en el dataset.
    geometrias = gpd.read_file(carpeta_datos / 'Parcelas_Reto_AGC_CONJUNTO.shp')  # Leemos los polígonos de FIRA.
    if geometrias.crs is None:  # Comprobamos que la geometría tenga un sistema de coordenadas declarado.
        raise ValueError('El Shapefile no declara el sistema de coordenadas.')  # Detenemos medidas espaciales sin referencia geográfica.
    centroides = geometrias.to_crs('EPSG:6372').centroid.to_crs('EPSG:4326')  # Calculamos centros en una proyección métrica y regresamos a coordenadas geográficas.
    geometrias['latitud_centro'] = centroides.y  # Registramos la latitud del centro de cada polígono.
    geometrias['longitud_centro'] = centroides.x  # Registramos la longitud del centro de cada polígono.
    geometrias = geometrias.rename(columns={'ID_POLIGON': 'ID_POLIGONO', 'Estado': 'estado', 'Municipio': 'municipio', 'CONJUNTO': 'CONJUNTO_MAPA', 'área_ha': 'AREA_MAPA_HA'})  # Normalizamos los nombres originales sin alterar los documentos.
    columnas = ['ID_POLIGONO', 'estado', 'municipio', 'CONJUNTO_MAPA', 'AREA_MAPA_HA', 'latitud_centro', 'longitud_centro']  # Enumeramos los atributos geográficos que vamos a revisar.
    if not objetivo['ID_POLIGONO'].is_unique or not geometrias['ID_POLIGONO'].is_unique:  # Comprobamos la unicidad de las parcelas en ambas fuentes.
        raise ValueError('Encontramos identificadores repetidos en las fuentes oficiales.')  # Evitamos duplicar parcelas al unir tablas.
    tabla = objetivo[['ID_POLIGONO', 'CONJUNTO', 'AREA_HA', 'RENDIMIENTO_T_HA']].merge(geometrias[columnas], on='ID_POLIGONO', validate='one_to_one')  # Asociamos exactamente una geometría a cada parcela.
    if len(tabla) != 197 or tabla['ID_POLIGONO'].nunique() != 197:  # Exigimos que la unión preserve las 197 parcelas.
        raise ValueError('La unión entre etiquetas y geometrías no conserva las 197 parcelas.')  # Alertamos si hay información geográfica perdida.
    if not tabla['CONJUNTO'].eq(tabla['CONJUNTO_MAPA']).all():  # Contrastamos las etiquetas de los dos archivos.
        raise ValueError('Las etiquetas de entrenamiento/predicción no coinciden entre fuentes.')  # Evitamos utilizar el grupo equivocado.
    if not np.allclose(tabla['AREA_HA'], tabla['AREA_MAPA_HA'], rtol=0, atol=0.011):  # Confirmamos que las superficies tabuladas coincidan dentro del redondeo.
        raise ValueError('La superficie declarada cambia entre geometría y tabla original.')  # Evitamos resultados económicos con áreas inconsistentes.
    if tabla['CONJUNTO'].eq('ENTRENAMIENTO').sum() != 138 or tabla['CONJUNTO'].eq('PREDICCION').sum() != 59:  # Confirmamos la separación oficial.
        raise ValueError('La partición de entrenamiento/predicción no tiene 138 y 59 parcelas.')  # Rechazamos particiones incompletas.
    return tabla.sort_values('ID_POLIGONO').reset_index(drop=True)  # Devolvemos una tabla estable para nuestra auditoría.



# ==================== 5. COBERTURA GEOGRÁFICA DEL CONJUNTO OCULTO ====================
def calcular_cobertura(tabla: pd.DataFrame, predicciones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:  # Comparamos ubicaciones ocultas con las 138 parcelas de referencia.
    if 'ID_POLIGONO' not in predicciones or len(predicciones) != 59 or not predicciones['ID_POLIGONO'].is_unique:  # Exigimos exactamente una predicción por parcela oculta.
        raise ValueError('El CSV de la Fase 4 no contiene 59 parcelas únicas.')  # Rechazamos predicciones incompletas.
    entrenamiento = tabla.loc[tabla['CONJUNTO'].eq('ENTRENAMIENTO')].reset_index(drop=True)  # Aislamos parcelas cuyos rendimientos conocemos.
    ocultas = tabla.loc[tabla['CONJUNTO'].eq('PREDICCION')].reset_index(drop=True)  # Aislamos parcelas cuyos rendimientos desconoce el equipo.
    if set(predicciones['ID_POLIGONO']) != set(ocultas['ID_POLIGONO']):  # Contrastamos los identificadores con las parcelas oficiales.
        raise ValueError('Las predicciones no corresponden exactamente a las 59 parcelas ocultas.')  # Detenemos asociaciones equivocadas.
    datos_ocultos = ocultas.merge(predicciones[['ID_POLIGONO', 'AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'PRODUCCION_ESTIMADA_T']], on='ID_POLIGONO', suffixes=('', '_PRED'), validate='one_to_one')  # Incorporamos valores estimados sin utilizar rendimientos reales.
    if not np.allclose(datos_ocultos['AREA_HA'], datos_ocultos['AREA_HA_PRED'], atol=1e-8):  # Comprobamos que el CSV no tenga superficies intercambiadas.
        raise ValueError('La superficie del CSV de predicción no coincide con la del dataset oficial.')  # Advertimos un error de integridad entre archivos.
    if not np.allclose(datos_ocultos['PRODUCCION_ESTIMADA_T'], datos_ocultos['AREA_HA'] * datos_ocultos['RENDIMIENTO_PREDICHO_T_HA'], atol=1e-5):  # Verificamos la fórmula de producción total.
        raise ValueError('La producción total no coincide con rendimiento por hectárea multiplicado por superficie.')  # Rechazamos predicciones con unidades inconsistentes.
    coordenadas = ['latitud_centro', 'longitud_centro']  # Fijamos el orden que espera la fórmula de Haversine.
    dist_ocultas = distancias_haversine_km(datos_ocultos[coordenadas].to_numpy(), entrenamiento[coordenadas].to_numpy())  # Calculamos la distancia entre cada parcela oculta y todas las etiquetadas.
    orden = np.argsort(dist_ocultas, axis=1)  # Identificamos vecinos cercanos sin utilizar sus rendimientos.
    datos_ocultos['distancia_vecino_1_km'] = np.take_along_axis(dist_ocultas, orden[:, :1], axis=1).ravel()  # Registramos la distancia del primer vecino.
    datos_ocultos['distancia_vecino_3_km'] = np.take_along_axis(dist_ocultas, orden[:, 2:3], axis=1).ravel()  # Registramos la distancia hasta el tercer vecino.
    datos_ocultos['id_vecino_1'] = entrenamiento['ID_POLIGONO'].to_numpy()[orden[:, 0]]  # Registramos el identificador del vecino más cercano.
    datos_ocultos['municipio_vecino_1'] = entrenamiento['municipio'].to_numpy()[orden[:, 0]]  # Identificamos el municipio del vecino de referencia.
    datos_ocultos['estado_vecino_1'] = entrenamiento['estado'].to_numpy()[orden[:, 0]]  # Identificamos el estado del vecino de referencia.
    dist_etiquetadas = distancias_haversine_km(entrenamiento[coordenadas].to_numpy(), entrenamiento[coordenadas].to_numpy())  # Calculamos distancias entre ejemplos etiquetados.
    np.fill_diagonal(dist_etiquetadas, np.inf)  # Excluimos cada parcela de su propia comparación de cercanía.
    entrenamiento['distancia_referencia_loo_km'] = np.min(dist_etiquetadas, axis=1)  # Obtenemos su vecino distinto más cercano como referencia.
    umbral_95 = float(np.quantile(entrenamiento['distancia_referencia_loo_km'], 0.95))  # Calculamos un umbral descriptivo con el 95% de las parcelas conocidas.
    datos_ocultos['supera_p95_distancia'] = datos_ocultos['distancia_vecino_1_km'] > umbral_95  # Señalamos ubicaciones más aisladas que nuestra referencia.
    datos_ocultos['vecino_del_mismo_municipio'] = datos_ocultos['municipio'].eq(datos_ocultos['municipio_vecino_1'])  # Verificamos correspondencia municipal del vecino más cercano.
    municipios_entrenados = set(entrenamiento['municipio'])  # Consultamos los municipios representados en entrenamiento.
    datos_ocultos['municipio_representado'] = datos_ocultos['municipio'].isin(municipios_entrenados)  # Marcamos municipios ausentes si los hubiera.
    envolventes_estado = {estado: MultiPoint(list(zip(grupo['longitud_centro'], grupo['latitud_centro']))).convex_hull for estado, grupo in entrenamiento.groupby('estado')}  # Construimos polígonos que envuelven los centros conocidos de cada estado.
    datos_ocultos['dentro_envolvente_estado'] = datos_ocultos.apply(lambda fila: bool(envolventes_estado[fila['estado']].covers(Point(fila['longitud_centro'], fila['latitud_centro']))), axis=1)  # Distinguimos puntos interiores de puntos situados fuera del contorno.
    limites_por_estado = entrenamiento.groupby('estado')['RENDIMIENTO_T_HA'].agg(['min', 'max'])  # Resumimos los límites observados sin suponer que sean límites físicos.
    datos_ocultos['dentro_rango_observado_estado'] = datos_ocultos.apply(lambda fila: bool(limites_por_estado.loc[fila['estado'], 'min'] <= fila['RENDIMIENTO_PREDICHO_T_HA'] <= limites_por_estado.loc[fila['estado'], 'max']), axis=1)  # Detectamos estimaciones fuera del rango observado localmente.
    grupo = datos_ocultos.groupby('estado').agg(parcelas=('ID_POLIGONO', 'count'), vecino_mediana_km=('distancia_vecino_1_km', 'median'), vecino_max_km=('distancia_vecino_1_km', 'max'), rendimiento_medio_estimado_t_ha=('RENDIMIENTO_PREDICHO_T_HA', 'mean'), fuera_envolvente=('dentro_envolvente_estado', lambda valores: int((~valores).sum())))  # Resumimos riesgos geográficos por estado.
    resumen = {'parcelas_etiquetadas': 138, 'parcelas_ocultas': 59, 'municipios_ocultos_representados': int(datos_ocultos['municipio_representado'].sum()), 'vecino_mismo_municipio': int(datos_ocultos['vecino_del_mismo_municipio'].sum()), 'distancia_mediana_km': float(datos_ocultos['distancia_vecino_1_km'].median()), 'distancia_max_km': float(datos_ocultos['distancia_vecino_1_km'].max()), 'referencia_p95_entrenamiento_km': umbral_95, 'parcelas_mas_aisladas_que_p95': int(datos_ocultos['supera_p95_distancia'].sum()), 'ocultas_dentro_envolvente_estado': int(datos_ocultos['dentro_envolvente_estado'].sum()), 'ocultas_fuera_envolvente_estado': int((~datos_ocultos['dentro_envolvente_estado']).sum()), 'predicciones_en_rango_estatal_observado': int(datos_ocultos['dentro_rango_observado_estado'].sum()), 'advertencia': 'La proximidad geográfica no prueba similitud agronómica ni exactitud de las predicciones.'}  # Guardamos métricas verificadas que podemos comunicar al equipo.
    return datos_ocultos, grupo.reset_index(), resumen  # Devolvemos las tablas y los indicadores para exportarlos.



# ==================== 6. ANÁLISIS DE ERRORES FUERA DE MUESTRA ====================
def calcular_sesgo_por_rendimiento(archivo_oof: Path) -> pd.DataFrame:  # Estudiamos si el modelo tiende a suavizar rendimientos extremos.
    tabla = pd.read_csv(archivo_oof)  # Recuperamos predicciones hechas sobre parcelas reservadas para validación.
    necesarios = {'ID_POLIGONO', 'real_t_ha', 'predicho_t_ha'}  # Definimos las columnas necesarias para calcular los errores.
    if not necesarios.issubset(tabla.columns) or len(tabla) != 138 or not tabla['ID_POLIGONO'].is_unique:  # Comprobamos que los errores correspondan una sola vez a cada parcela etiquetada.
        raise ValueError('La tabla de validación anidada no contiene las 138 predicciones esperadas.')  # Evitamos evaluar filas duplicadas o incompletas.
    tabla['grupo_rendimiento'] = pd.qcut(tabla['real_t_ha'], 4, duplicates='drop').astype(str)  # Separamos las parcelas por cuartiles de su rendimiento real.
    tabla['sesgo_predicho_menos_real'] = tabla['predicho_t_ha'] - tabla['real_t_ha']  # Calculamos el signo del error de cada predicción.
    tabla['error_absoluto'] = tabla['sesgo_predicho_menos_real'].abs()  # Medimos cuánto se equivocó cada parcela sin atender al signo.
    grupos = tabla.groupby('grupo_rendimiento', as_index=False).agg(parcelas=('ID_POLIGONO', 'count'), rendimiento_real_medio=('real_t_ha', 'mean'), rendimiento_estimado_medio=('predicho_t_ha', 'mean'), sesgo_promedio_t_ha=('sesgo_predicho_menos_real', 'mean'), mae_t_ha=('error_absoluto', 'mean'))  # Resumimos los errores de cada grupo de rendimiento.
    return grupos  # Entregamos una tabla útil para explicar riesgos y limitaciones en el informe técnico.



# ==================== 7. GRÁFICAS PARA EL INFORME Y LA EXPOSICIÓN ====================
def guardar_graficas(ocultas: pd.DataFrame, resumen: dict, salida: Path) -> None:  # Visualizamos las observaciones que sustentan nuestras conclusiones.
    fig, eje = plt.subplots(figsize=(9, 5))  # Creamos una figura con espacio suficiente para leer las etiquetas.
    orden = ocultas.sort_values('distancia_vecino_1_km', ascending=False)  # Ordenamos las parcelas desde la más alejada.
    eje.bar(range(len(orden)), orden['distancia_vecino_1_km'].to_numpy())  # Mostramos las distancias de las 59 parcelas.
    eje.axhline(resumen['referencia_p95_entrenamiento_km'], linestyle='--', label='Percentil 95 de referencia')  # Añadimos el umbral obtenido de las parcelas etiquetadas.
    eje.set(title='Distancia al entrenamiento más cercano', xlabel='59 parcelas ocultas, ordenadas por distancia', ylabel='Distancia entre centroides (km)')  # Explicamos qué representan los ejes y sus unidades.
    eje.legend()  # Identificamos la línea de referencia.
    fig.tight_layout()  # Evitamos que el texto quede fuera de la figura.
    fig.savefig(salida / 'distancias_59_parcelas.png', dpi=180)  # Guardamos la visualización en la carpeta de auditoría.
    plt.close(fig)  # Liberamos recursos para la siguiente figura.
    fig, eje = plt.subplots(figsize=(8, 5))  # Preparamos la comparación de predicciones entre estados.
    datos = [grupo['RENDIMIENTO_PREDICHO_T_HA'].to_numpy() for _, grupo in ocultas.groupby('estado', sort=True)]  # Separamos rendimientos estimados por ubicación.
    etiquetas = [nombre for nombre, _ in ocultas.groupby('estado', sort=True)]  # Recuperamos las etiquetas de cada estado.
    eje.boxplot(datos, tick_labels=etiquetas)  # Mostramos mediana y dispersión de las predicciones geográficas.
    eje.set(title='Predicciones candidatas por estado', ylabel='Rendimiento estimado (t/ha)', xlabel='Estado')  # Etiquetamos la gráfica con las unidades correctas.
    fig.tight_layout()  # Adaptamos márgenes para que nada se recorte.
    fig.savefig(salida / 'predicciones_por_estado.png', dpi=180)  # Guardamos la figura sin presentarla como rendimiento real.
    plt.close(fig)  # Cerramos la figura para liberar memoria.



# ==================== 7B. MAPA DESCRIPTIVO DE COBERTURA ====================
def guardar_mapa_cobertura(parcelas: pd.DataFrame, auditoria: pd.DataFrame, salida: Path) -> None:  # Mostramos las parcelas conocidas y las ocultas sin utilizar rendimientos no disponibles.
    etiquetas = parcelas.loc[parcelas['CONJUNTO'].eq('ENTRENAMIENTO')]  # Separamos las ubicaciones utilizadas durante el entrenamiento.
    fig, eje = plt.subplots(figsize=(9, 7))  # Preparamos una imagen grande para identificar agrupaciones espaciales.
    eje.scatter(etiquetas['longitud_centro'], etiquetas['latitud_centro'], marker='o', s=27, alpha=0.58, label='Entrenamiento (138)')  # Mostramos las coordenadas centrales de las parcelas conocidas.
    eje.scatter(auditoria['longitud_centro'], auditoria['latitud_centro'], marker='^', s=47, alpha=0.9, label='Predicción (59)')  # Diferenciamos las ubicaciones que debemos predecir.
    exteriores = auditoria.loc[~auditoria['dentro_envolvente_estado']]  # Identificamos las ubicaciones exteriores a los contornos estatales.
    eje.scatter(exteriores['longitud_centro'], exteriores['latitud_centro'], marker='s', facecolors='none', edgecolors='black', s=85, linewidths=1.2, label='Fuera de envolvente estatal (10)')  # Marcamos los casos que requieren una interpretación cuidadosa.
    eje.set(title='Cobertura geográfica: entrenamiento y 59 parcelas ocultas', xlabel='Longitud (grados)', ylabel='Latitud (grados)')  # Explicamos las coordenadas y el propósito del gráfico.
    eje.set_aspect(1 / np.cos(np.radians(float(parcelas['latitud_centro'].mean()))))  # Ajustamos visualmente la relación de escala entre grados de latitud y longitud.
    eje.legend(loc='best')  # Aclaramos el significado de cada símbolo utilizado.
    fig.tight_layout()  # Ajustamos los márgenes para evitar recortar etiquetas.
    fig.savefig(salida / 'mapa_cobertura_centroides.png', dpi=180)  # Guardamos una figura descriptiva, no un mapa catastral de precisión.
    plt.close(fig)  # Liberamos los recursos de la figura una vez exportada.



# ==================== 8. EJECUCIÓN CONTROLADA Y EXPORTACIÓN ====================
def ejecutar(datos: Path, predicciones: Path, fase3: Path, salida: Path) -> dict:  # Ejecutamos una auditoría descriptiva separada del entrenamiento.
    datos = datos.resolve()  # Normalizamos la ruta del dataset original.
    salida = salida.resolve()  # Normalizamos la ruta de los resultados nuevos.
    if salida == datos or datos in salida.parents:  # Rechazamos salidas que puedan modificar los documentos oficiales.
        raise ValueError('No guardamos resultados dentro de DATASET_RETO_AGRO_2026.')  # Impedimos alterar los archivos de FIRA.
    if not predicciones.is_file():  # Verificamos que la Fase 4 haya generado su CSV candidato.
        raise FileNotFoundError(f'Primero ejecutamos la Fase 4; no encontramos {predicciones}')  # Informamos la dependencia sin usar datos inventados.
    tabla = cargar_parcelas(datos)  # Abrimos y verificamos las geometrías y el conjunto objetivo.
    estimaciones = pd.read_csv(predicciones)  # Recuperamos las 59 predicciones que ya generó Extra Trees.
    ocultas, por_estado, resumen = calcular_cobertura(tabla, estimaciones)  # Calculamos cobertura y estadísticas sin reentrenar.
    salida.mkdir(parents=True, exist_ok=True)  # Creamos una carpeta específica fuera del dataset.
    columnas = ['ID_POLIGONO', 'estado', 'municipio', 'AREA_HA', 'RENDIMIENTO_PREDICHO_T_HA', 'distancia_vecino_1_km', 'distancia_vecino_3_km', 'id_vecino_1', 'municipio_vecino_1', 'municipio_representado', 'vecino_del_mismo_municipio', 'supera_p95_distancia', 'dentro_envolvente_estado', 'dentro_rango_observado_estado']  # Elegimos el conjunto legible de campos de auditoría.
    ocultas[columnas].sort_values('ID_POLIGONO').to_csv(salida / 'auditoria_59_parcelas.csv', index=False, float_format='%.6f')  # Guardamos 59 filas sin alterar el CSV original de predicción.
    por_estado.to_csv(salida / 'auditoria_resumen_por_estado.csv', index=False, float_format='%.6f')  # Guardamos indicadores por estado para la reunión del equipo.
    if fase3.exists():  # Comprobamos si también podemos calcular sesgos con la Fase 3.
        sesgo = calcular_sesgo_por_rendimiento(fase3)  # Utilizamos solo predicciones fuera de muestra para cuantificar errores reales.
        sesgo.to_csv(salida / 'sesgos_por_nivel_rendimiento.csv', index=False, float_format='%.6f')  # Conservamos el resultado documentado de la prueba de sesgo.
        resumen['auditoria_sesgo_oof_disponible'] = True  # Informamos que verificamos la validación anidada.
    else:  # Permitimos una auditoría geográfica aun cuando no esté disponible el archivo opcional de la Fase 3.
        resumen['auditoria_sesgo_oof_disponible'] = False  # Advertimos de manera transparente que falta el análisis de sesgo.
    guardar_graficas(ocultas, resumen, salida)  # Creamos dos figuras de apoyo para la exposición.
    guardar_mapa_cobertura(tabla, ocultas, salida)  # Mostramos también la posición relativa de las parcelas conocidas y ocultas.
    (salida / 'resumen_auditoria.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')  # Guardamos valores estructurados para repetir y comparar la auditoría.
    print('AUDITORÍA GEOGRÁFICA TERMINADA')  # Informamos que los cálculos concluyeron.
    print(json.dumps(resumen, ensure_ascii=False, indent=2))  # Mostramos los resultados principales en la terminal.
    print('Archivos generados en:', salida)  # Indicamos dónde se encuentran todas las salidas.
    print('ESTADO: revisión científica; no modifica ni aprueba automáticamente las predicciones oficiales.')  # Distinguimos auditoría de aprobación o entrega.
    return resumen  # Devolvemos los indicadores para que puedan comprobarse en pruebas.



# ==================== 9. ENTRADA DESDE LA TERMINAL ====================
def main() -> None:  # Permitimos ejecutar la auditoría con rutas portables.
    argumentos = argparse.ArgumentParser(description='AgroBYTE: auditoría de cobertura de las 59 predicciones FIRA')  # Documentamos el propósito del módulo.
    argumentos.add_argument('--datos', type=Path, default=DATOS_PREDETERMINADOS)  # Permitimos sustituir únicamente la fuente de lectura.
    argumentos.add_argument('--predicciones', type=Path, default=PREDICCIONES_PREDETERMINADAS)  # Permitimos elegir el CSV de Fase 4 que se auditará.
    argumentos.add_argument('--fase3', type=Path, default=RAIZ / 'salidas' / 'fase3' / 'resultados' / 'predicciones_anidadas_fase3.csv')  # Buscamos los errores de validación de la Fase 3 si existen.
    argumentos.add_argument('--salida', type=Path, default=SALIDA_PREDETERMINADA)  # Separamos todos los archivos nuevos del dataset original.
    opciones = argumentos.parse_args()  # Recuperamos los argumentos introducidos en la terminal.
    ejecutar(opciones.datos, opciones.predicciones, opciones.fase3, opciones.salida)  # Ejecutamos los controles con las rutas seleccionadas.


if __name__ == '__main__':  # Evitamos ejecutar la auditoría al importarla desde un archivo de pruebas.
    main()  # Iniciamos el proceso cuando ejecutamos el script directamente.
