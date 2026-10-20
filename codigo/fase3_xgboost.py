"""Fase 3 de AgroBYTE: evaluación rigurosa de XGBoost frente a nuestros modelos previos.

El archivo no modifica las fases 1 y 2. Todos los resultados son EXPLORATORIOS.
"""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Habilitamos anotaciones modernas de tipos.
import argparse  # Leemos opciones escritas por el equipo en la terminal.
import json  # Registramos las configuraciones para reproducir los experimentos.
import platform  # Documentamos el entorno informático de cada ejecución.
import sys  # Localizamos los módulos compartidos del proyecto.
from datetime import datetime, timezone  # Registramos cuándo se obtuvieron los resultados.
from pathlib import Path  # Construimos rutas válidas en distintas computadoras.
import joblib  # Guardamos un modelo experimental con sus transformaciones.
import matplotlib  # Preparamos gráficas que pueden generarse sin pantalla.
matplotlib.use('Agg')  # Permitimos que el experimento funcione también en servidores.
import matplotlib.pyplot as plt  # Elaboramos gráficas para interpretar los errores.
import numpy as np  # Calculamos predicciones y estadísticas numéricas.
import pandas as pd  # Organizamos resultados en tablas y archivos CSV.
import sklearn  # Documentamos la versión de la biblioteca de validación.
import xgboost  # Documentamos la versión de XGBoost realmente utilizada.
from xgboost import XGBRegressor  # Incorporamos un nuevo algoritmo de regresión con boosting.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Calculamos las métricas del concurso.
from sklearn.model_selection import GroupKFold, StratifiedKFold  # Separamos parcelas o municipios según la pregunta científica.


# ==================== 2. REUTILIZACIÓN DE LA FASE 2 ====================
RAIZ = Path(__file__).resolve().parent.parent  # Identificamos la raíz del proyecto sin rutas personales.
sys.path.insert(0, str(RAIZ / 'codigo'))  # Permitimos importar las funciones que ya comprobamos en la Fase 2.
import fase2_modelos_comparacion as fase2  # Reutilizamos exactamente sus reglas de selección y preparación de variables.
from rutas import carpeta_dataset, validar_carpeta_salida  # Protegemos los datos originales de FIRA.


# ==================== 3. CONFIGURACIONES FIJADAS ANTES DE COMPARAR ====================
SEMILLAS = (42, 73, 2026)  # Repetimos las mismas semillas utilizadas en la Fase 2.
CANDIDATOS = (  # Declaramos los candidatos antes de conocer sus errores.
    ('geo_extratrees', 'geografia', 'extratrees'),  # Conservamos el ganador exploratorio previo como referencia.
    ('reducido_catboost', 'reducido', 'catboost'),  # Conservamos el candidato que mejor resistió municipios no vistos.
    ('reducido_xgboost_conservador', 'reducido', 'xgb_conservador'),  # Probamos boosting suave con pocos grados de libertad.
    ('completo_xgboost_conservador', 'completo', 'xgb_conservador'),  # Comprobamos si los índices adicionales aportan información.
    ('reducido_xgboost_alternativo', 'reducido', 'xgb_alternativo'),  # Contrastamos una segunda regularización moderada.
)  # Cerramos las hipótesis fijadas previamente.
SELECCION_ANIDADA = ('geo_extratrees', 'reducido_catboost', 'reducido_xgboost_conservador', 'completo_xgboost_conservador')  # Limitamos el número de modelos elegibles internamente.


# ==================== 4. CREACIÓN DE LOS MODELOS ====================
def construir_candidato(nombre, variables):  # Construimos desde cero un pipeline para cada partición.
    if nombre in ('extratrees', 'catboost'):  # Identificamos algoritmos ya probados en la Fase 2.
        return fase2.construir_modelo(nombre, variables)  # Conservamos exactamente su preprocesamiento y sus hiperparámetros.
    if nombre not in ('xgb_conservador', 'xgb_alternativo'):  # Rechazamos configuraciones inesperadas.
        raise ValueError(f'Algoritmo desconocido: {nombre}')  # Evitamos ejecutar experimentos con nombres equivocados.
    modelo = fase2.construir_modelo('extratrees', variables)  # Reutilizamos la misma codificación y la misma imputación que Extra Trees.
    if nombre == 'xgb_conservador':  # Elegimos la variante más regularizada, apropiada para pocos ejemplos.
        opciones = dict(n_estimators=210, max_depth=2, learning_rate=0.035, min_child_weight=5, subsample=0.85, colsample_bytree=0.85, reg_lambda=12, reg_alpha=0.7, gamma=0.1)  # Reducimos el riesgo de sobreajuste.
    else:  # Identificamos la configuración XGBoost alternativa.
        opciones = dict(n_estimators=300, max_depth=2, learning_rate=0.025, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, reg_lambda=6, reg_alpha=0.3, gamma=0.0)  # Permitimos una flexibilidad ligeramente superior.
    regresor = XGBRegressor(objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=2, verbosity=0, **opciones)  # Construimos XGBoost con resultados repetibles.
    modelo.set_params(regresion=regresor)  # Sustituimos solamente el algoritmo y dejamos intacta la preparación de datos.
    return modelo  # Devolvemos el pipeline completo sin haberlo entrenado todavía.


# ==================== 5. CARGA Y VALIDACIÓN DE LAS 197 PARCELAS ====================
def leer_parcelas(datos, fase1):  # Recuperamos exactamente las matrices que produce el pipeline validado.
    matriz = Path(fase1) / 'caracteristicas_nubes_20.csv'  # Localizamos las variables calculadas con el umbral del 20%.
    if not matriz.exists():  # Comprobamos que la Fase 1 se haya ejecutado antes.
        raise FileNotFoundError('Falta caracteristicas_nubes_20.csv. Ejecutamos primero: python ejecutar.py fase1')  # Explicamos qué debe corregirse.
    fase2.BASE = Path(datos)  # Indicamos al módulo compartido dónde están los originales de FIRA.
    fase2.PREV = Path(fase1)  # Indicamos al módulo compartido dónde están las matrices ya preparadas.
    entrenamiento, ocultas = fase2.cargar_datos()  # Reutilizamos sus comprobaciones de integridad y separación de etiquetas.
    assert len(entrenamiento) == 138 and len(ocultas) == 59  # Impedimos perder parcelas o duplicarlas sin darnos cuenta.
    assert entrenamiento['ID_POLIGONO'].is_unique and ocultas['ID_POLIGONO'].is_unique  # Exigimos identificadores únicos.
    assert ocultas['RENDIMIENTO_T_HA'].isna().all()  # Confirmamos que nunca contamos con las respuestas de FIRA.
    return entrenamiento, ocultas  # Conservamos por separado las parcelas etiquetadas y las ocultas.


# ==================== 6. CÁLCULO DE MÉTRICAS ====================
def calcular_metricas(reales, estimadas):  # Evaluamos los mismos indicadores para todos los algoritmos.
    reales = np.asarray(reales, dtype=float)  # Convertimos los rendimientos observados en números.
    estimadas = np.asarray(estimadas, dtype=float)  # Convertimos las predicciones en números.
    return {'rmse': float(np.sqrt(mean_squared_error(reales, estimadas))), 'mae': float(mean_absolute_error(reales, estimadas)), 'r2': float(r2_score(reales, estimadas))}  # Calculamos RMSE, MAE y R² sin mezclarlos con datos ocultos.


# ==================== 7. EVALUACIÓN DE UNA CONFIGURACIÓN ====================
def predecir_fuera_de_muestra(entrenamiento, definicion, particiones, semilla, protocolo):  # Probamos parcelas que nunca participaron en el ajuste de su propio fold.
    nombre, grupo, algoritmo = definicion  # Recuperamos el nombre, el conjunto de columnas y la familia de modelo.
    columnas = fase2.columnas_para(entrenamiento, grupo)  # Usamos las mismas familias de variables de la Fase 2.
    X = entrenamiento[columnas]  # Excluimos el identificador, la etiqueta y la bandera de evaluación.
    y = entrenamiento['RENDIMIENTO_T_HA'].to_numpy(dtype=float)  # Trabajamos exclusivamente con los 138 rendimientos conocidos.
    oof = np.full(len(y), np.nan)  # Reservamos un pronóstico externo para cada parcela etiquetada.
    pliegues = np.zeros(len(y), dtype=int)  # Guardamos la partición donde se evaluó cada parcela.
    for numero, (indices_ajuste, indices_prueba) in enumerate(particiones, 1):  # Recorremos divisiones previamente fijadas para evitar comparaciones desiguales.
        modelo = construir_candidato(algoritmo, X)  # Construimos un nuevo pipeline vacío para impedir fugas entre folds.
        modelo.fit(X.iloc[indices_ajuste], y[indices_ajuste])  # Ajustamos imputación, codificación y árbol únicamente con entrenamiento.
        oof[indices_prueba] = modelo.predict(X.iloc[indices_prueba])  # Estimamos rendimientos de parcelas reservadas sin mostrar sus etiquetas.
        pliegues[indices_prueba] = numero  # Conservamos la identidad de la partición usada en esta predicción.
    if not np.isfinite(oof).all() or not (pliegues > 0).all():  # Revisamos la cobertura y validez de todas las predicciones.
        raise RuntimeError(f'Faltan predicciones en {nombre}, {protocolo}, semilla {semilla}')  # Evitamos reportar una evaluación incompleta.
    detalle = entrenamiento[['ID_POLIGONO', 'estado', 'municipio']].copy()  # Conservamos la geografía de cada registro para los diagnósticos.
    detalle['real_t_ha'] = y  # Registramos el rendimiento real usado exclusivamente para evaluación.
    detalle['predicho_t_ha'] = oof  # Registramos el resultado obtenido sin ver cada rendimiento durante el entrenamiento.
    detalle['error_t_ha'] = oof - y  # Calculamos el error firmado, útil para detectar sesgos regionales.
    detalle['fold'] = pliegues  # Indicamos la partición que produjo cada predicción.
    detalle['semilla'] = semilla  # Identificamos la división aleatoria usada para reproducir la prueba.
    detalle['protocolo'] = protocolo  # Distinguimos validación por parcela frente a la espacial.
    detalle['candidato'] = nombre  # Indicamos qué algoritmo produjo los resultados.
    return calcular_metricas(y, oof), detalle  # Devolvemos métricas y errores individuales para auditoría.


# ==================== 8. VALIDACIÓN CRUZADA REPETIDA POR PARCELAS ====================
def comparar_parcelas(entrenamiento, rapido=False):  # Comparamos configuraciones con las mismas divisiones y tres semillas.
    filas, detalles = [], []  # Acumulamos resúmenes y registros individuales sin sobrescribir resultados.
    semillas = SEMILLAS[:1] if rapido else SEMILLAS  # Acortamos solamente el número de repeticiones al solicitar una prueba rápida.
    for semilla in semillas:  # Repetimos el proceso para examinar estabilidad de los errores.
        cortes = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla).split(entrenamiento, entrenamiento['estado']))  # Garantizamos presencia regional en cada partición.
        for candidato in CANDIDATOS:  # Utilizamos idénticos folds para los modelos nuevos y anteriores.
            metricas, detalle = predecir_fuera_de_muestra(entrenamiento, candidato, cortes, semilla, 'parcelas')  # Entrenamos sin tocar las 59 etiquetas ocultas.
            filas.append({'semilla': semilla, 'candidato': candidato[0], 'variables': candidato[1], 'algoritmo': candidato[2], **metricas})  # Conservamos resultados completos del experimento.
            detalles.append(detalle)  # Guardamos errores individuales para comprender las diferencias.
            print(f'PARCELAS {semilla} {candidato[0]} RMSE={metricas["rmse"]:.5f}', flush=True)  # Informamos el avance real de la validación.
    return pd.DataFrame(filas), pd.concat(detalles, ignore_index=True)  # Construimos tablas reutilizables para el reporte técnico.


# ==================== 9. VALIDACIÓN POR MUNICIPIOS NO VISTOS ====================
def comparar_municipios(entrenamiento):  # Medimos qué sucede al excluir municipios enteros del entrenamiento.
    grupos = entrenamiento['estado'].astype(str) + '/' + entrenamiento['municipio'].astype(str)  # Definimos los grupos geográficos sin mezclar municipios homónimos.
    cortes = list(GroupKFold(n_splits=5).split(entrenamiento, groups=grupos))  # Impedimos que un municipio aparezca en ambos lados de una partición.
    filas, detalles = [], []  # Preparamos registros para cada candidato y cada parcela.
    for candidato in CANDIDATOS:  # Probamos los mismos algoritmos con un criterio espacial más exigente.
        metricas, detalle = predecir_fuera_de_muestra(entrenamiento, candidato, cortes, 0, 'municipios')  # Generamos resultados externos a los municipios del entrenamiento.
        filas.append({'candidato': candidato[0], **metricas})  # Resumimos la calidad de la extrapolación geográfica.
        detalles.append(detalle)  # Preservamos todos los errores individuales para diagnósticos.
        print(f'MUNICIPIOS {candidato[0]} RMSE={metricas["rmse"]:.5f}', flush=True)  # Mostramos los resultados sin seleccionar prematuramente un ganador.
    return pd.DataFrame(filas), pd.concat(detalles, ignore_index=True)  # Retornamos métricas y evidencias completas.


# ==================== 10. SELECCIÓN ANIDADA CON EVALUACIÓN EXTERNA ====================
def seleccion_anidada(entrenamiento):  # Separamos la elección de un modelo de la evaluación del modelo seleccionado.
    cortes_externos = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=319).split(entrenamiento, entrenamiento['estado']))  # Reutilizamos la semilla externa de la Fase 2.
    definiciones = [candidato for candidato in CANDIDATOS if candidato[0] in SELECCION_ANIDADA]  # Limitamos la búsqueda interna a cuatro opciones predefinidas.
    valores_y = entrenamiento['RENDIMIENTO_T_HA'].to_numpy(dtype=float)  # Leemos las etiquetas sin incluir jamás las 59 de evaluación oficial.
    estimadas = np.full(len(entrenamiento), np.nan)  # Reservamos una estimación externa por parcela etiquetada.
    elecciones = []  # Conservamos la selección realizada dentro de cada entrenamiento externo.
    for numero, (indices_ajuste, indices_prueba) in enumerate(cortes_externos, 1):  # Apartamos el fold externo antes de decidir el algoritmo.
        interior = entrenamiento.iloc[indices_ajuste].reset_index(drop=True)  # Limitamos toda la búsqueda al conjunto permitido.
        cortes_internos = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=100 + numero).split(interior, interior['estado']))  # Definimos tres pliegues internos con presencia regional.
        candidatos_internos = []  # Preparamos los resultados que definirán el modelo elegido.
        for candidato in definiciones:  # Comparamos únicamente con rendimientos permitidos en la parte interna.
            metricas, _ = predecir_fuera_de_muestra(interior, candidato, cortes_internos, numero, 'interno')  # Obtenemos RMSE interno de cada opción.
            candidatos_internos.append((metricas['rmse'], candidato))  # Asociamos el error de cada algoritmo con su configuración.
        mejor_rmse, ganador = min(candidatos_internos, key=lambda registro: registro[0])  # Elegimos por RMSE sin consultar el fold externo.
        columnas = fase2.columnas_para(entrenamiento, ganador[1])  # Recuperamos la familia de variables del ganador del fold.
        X = entrenamiento[columnas]  # Construimos una matriz común con características permitidas.
        modelo = construir_candidato(ganador[2], X)  # Inicializamos un estimador completamente nuevo.
        modelo.fit(X.iloc[indices_ajuste], valores_y[indices_ajuste])  # Ajustamos el modelo con el entrenamiento externo completo.
        estimadas[indices_prueba] = modelo.predict(X.iloc[indices_prueba])  # Evaluamos el fold externo reservado desde el principio.
        elecciones.append({'fold_externo': numero, 'candidato_elegido': ganador[0], 'rmse_interno': mejor_rmse, 'parcelas_validacion_externa': len(indices_prueba)})  # Documentamos la elección sin confundir sus errores.
        print(f'ANIDADA fold {numero}: {ganador[0]} (RMSE interno {mejor_rmse:.5f})', flush=True)  # Dejamos un seguimiento claro del cálculo.
    if not np.isfinite(estimadas).all():  # Comprobamos que cada parcela recibió una única predicción externa.
        raise RuntimeError('La validación anidada no produjo las 138 predicciones externas.')  # Rechazamos métricas basadas en resultados incompletos.
    detalle = entrenamiento[['ID_POLIGONO', 'estado', 'municipio']].copy()  # Conservamos las claves necesarias para revisar los errores.
    detalle['real_t_ha'] = valores_y  # Guardamos los rendimientos observados para la evaluación externa.
    detalle['predicho_t_ha'] = estimadas  # Guardamos las predicciones que nunca vieron su propia etiqueta.
    detalle['error_t_ha'] = estimadas - valores_y  # Calculamos el sesgo firmado por parcela.
    return calcular_metricas(valores_y, estimadas), detalle, pd.DataFrame(elecciones)  # Entregamos una evaluación completa y auditable.


# ==================== 11. COMPARACIONES Y GRÁFICAS ====================
def crear_graficas(comparacion, espacial, detalle, salida):  # Convertimos los números en figuras comprensibles para el jurado.
    ranking = comparacion.groupby('candidato', as_index=False).agg(rmse_promedio=('rmse', 'mean'), rmse_desviacion=('rmse', 'std')).sort_values('rmse_promedio')  # Resumimos desempeño e inestabilidad entre semillas.
    grafica, eje = plt.subplots(figsize=(11, 6))  # Preparamos una gráfica horizontal apta para una presentación.
    eje.barh(ranking['candidato'], ranking['rmse_promedio'], xerr=ranking['rmse_desviacion'].fillna(0), capsize=3)  # Comparamos errores mediante barras con dispersión.
    eje.invert_yaxis()  # Colocamos las barras de menor error en primer lugar.
    eje.set_xlabel('RMSE (toneladas por hectárea)')  # Etiquetamos las unidades originales del rendimiento.
    eje.set_title('AgroBYTE — Fase 3: comparación exploratoria de modelos')  # Identificamos esta figura como experimento.
    grafica.tight_layout()  # Corregimos márgenes para evitar nombres cortados.
    grafica.savefig(salida / 'rmse_comparacion_fase3.png', dpi=160)  # Exportamos una figura reutilizable por el equipo.
    plt.close(grafica)  # Liberamos memoria después de escribir el archivo.
    grafica, eje = plt.subplots(figsize=(8, 7))  # Creamos otra figura para mostrar predicciones frente a observaciones.
    eje.scatter(detalle['real_t_ha'], detalle['predicho_t_ha'], alpha=0.7, s=28)  # Dibujamos cada parcela con sus valores conocidos y estimados.
    minimo = min(float(detalle['real_t_ha'].min()), float(detalle['predicho_t_ha'].min()))  # Localizamos el extremo inferior de los ejes.
    maximo = max(float(detalle['real_t_ha'].max()), float(detalle['predicho_t_ha'].max()))  # Localizamos el extremo superior de los ejes.
    eje.plot([minimo, maximo], [minimo, maximo], '--', linewidth=1)  # Dibujamos la diagonal de las predicciones perfectas.
    eje.set_xlabel('Rendimiento real (t/ha)')  # Identificamos los valores conocidos del dataset.
    eje.set_ylabel('Rendimiento predicho (t/ha)')  # Identificamos las salidas del modelo de evaluación.
    eje.set_title('AgroBYTE — Rendimiento real vs. predicho (validación)')  # Aclaramos que no usamos el conjunto oculto.
    grafica.tight_layout()  # Ajustamos el espacio para que la figura sea legible.
    grafica.savefig(salida / 'real_vs_predicho_fase3.png', dpi=160)  # Guardamos una segunda figura para documentar errores.
    plt.close(grafica)  # Cerramos la figura tras su exportación.
    grafica, eje = plt.subplots(figsize=(10, 5))  # Abrimos un gráfico para la prueba de municipios.
    orden = espacial.sort_values('rmse')  # Ordenamos algoritmos según su generalización espacial.
    eje.barh(orden['candidato'], orden['rmse'])  # Representamos los errores fuera de municipio.
    eje.invert_yaxis()  # Situamos el RMSE espacial más pequeño arriba.
    eje.set_xlabel('RMSE por municipios excluidos (t/ha)')  # Describimos qué evaluación se está visualizando.
    eje.set_title('AgroBYTE — Generalización hacia municipios no vistos')  # Evitamos confundir esta prueba con la CV por parcelas.
    grafica.tight_layout()  # Dejamos márgenes suficientes para etiquetas.
    grafica.savefig(salida / 'rmse_espacial_fase3.png', dpi=160)  # Exportamos la tercera figura.
    plt.close(grafica)  # Liberamos recursos de la gráfica final.


# ==================== 12. EXPORTACIÓN PROVISIONAL SIN ENTREGAR A FIRA ====================
def exportar_borrador(entrenamiento, ocultas, ganador, salida):  # Guardamos una predicción experimental que aún no es la entrega oficial.
    definicion = next(candidato for candidato in CANDIDATOS if candidato[0] == ganador)  # Recuperamos la configuración exploratoria seleccionada.
    columnas = fase2.columnas_para(entrenamiento, definicion[1])  # Seleccionamos las características correspondientes al mismo modelo.
    X = entrenamiento[columnas]  # Extraemos las variables con las que podemos entrenar.
    destino = ocultas[columnas]  # Construimos exactamente las mismas variables para las 59 parcelas ocultas.
    modelo = construir_candidato(definicion[2], X)  # Creamos una instancia sin parámetros aprendidos.
    modelo.fit(X, entrenamiento['RENDIMIENTO_T_HA'].to_numpy(float))  # Entrenamos con los 138 rendimientos visibles.
    valores = modelo.predict(destino)  # Estimamos rendimientos sin consultar ninguna respuesta real oculta.
    if len(valores) != 59 or not np.isfinite(valores).all():  # Comprobamos que no existan predicciones faltantes.
        raise ValueError('El borrador no contiene exactamente 59 predicciones válidas.')  # Evitamos exportar resultados incompletos.
    tabla = ocultas[['ID_POLIGONO', 'AREA_HA', 'estado', 'municipio']].copy()  # Conservamos los identificadores para relacionar resultados con el mapa.
    tabla['RENDIMIENTO_PREDICHO_T_HA'] = valores  # Expresamos cada predicción en toneladas por hectárea.
    tabla['PRODUCCION_ESTIMADA_T'] = valores * tabla['AREA_HA'].to_numpy(float)  # Calculamos la producción total mediante superficie por rendimiento.
    tabla.to_csv(salida / 'predicciones_59_FASE3_EXPERIMENTALES_NO_ENTREGAR.csv', index=False)  # Señalamos que es un archivo experimental, no oficial.
    joblib.dump(modelo, salida.parent / 'modelos' / 'fase3_modelo_EXPERIMENTAL_NO_FINAL.joblib')  # Conservamos el modelo junto a su preprocesamiento.
    return tabla  # Devolvemos el borrador completo para revisar sus rangos.


# ==================== 13. ORQUESTACIÓN REPRODUCIBLE DEL EXPERIMENTO ====================
def main():  # Ejecutamos la comparación solamente cuando se solicita por terminal.
    opciones = argparse.ArgumentParser(description='AgroBYTE Fase 3: XGBoost, Extra Trees, CatBoost y validación rigurosa')  # Documentamos las opciones para el equipo.
    opciones.add_argument('--datos', type=Path, default=None)  # Permitimos especificar una ubicación alternativa del dataset.
    opciones.add_argument('--fase1', type=Path, default=RAIZ / 'salidas' / 'fase1')  # Reutilizamos la matriz creada en la Fase 1.
    opciones.add_argument('--salida', type=Path, default=RAIZ / 'salidas' / 'fase3')  # Aislamos completamente los resultados de esta nueva fase.
    opciones.add_argument('--rapido', action='store_true')  # Permitimos una prueba breve antes de la evaluación completa.
    args = opciones.parse_args()  # Interpretamos los argumentos proporcionados desde VS Code.
    datos = carpeta_dataset(args.datos)  # Localizamos los datos originales sin escribir dentro de ellos.
    salida = validar_carpeta_salida(datos, args.salida)  # Comprobamos que nunca guardaremos resultados en la carpeta FIRA.
    resultados = salida / 'resultados'  # Separamos tablas y gráficas de los modelos guardados.
    resultados.mkdir(parents=True, exist_ok=True)  # Creamos las carpetas necesarias si todavía no existen.
    (salida / 'modelos').mkdir(parents=True, exist_ok=True)  # Preparamos un destino exclusivo para modelos experimentales.
    entrenamiento, ocultas = leer_parcelas(datos, args.fase1)  # Cargamos los conjuntos sin revelar los 59 rendimientos ocultos.
    comparacion, detalle = comparar_parcelas(entrenamiento, args.rapido)  # Medimos cinco modelos bajo las mismas particiones.
    comparacion.to_csv(resultados / 'comparacion_cv_fase3.csv', index=False)  # Guardamos errores por candidato y semilla.
    detalle.to_csv(resultados / 'predicciones_cv_fase3.csv', index=False)  # Guardamos predicciones fuera de muestra para auditar errores.
    ranking = comparacion.groupby('candidato', as_index=False).agg(rmse_promedio=('rmse', 'mean'), rmse_desviacion=('rmse', 'std'), mae_promedio=('mae', 'mean'), r2_promedio=('r2', 'mean')).sort_values('rmse_promedio')  # Calculamos una clasificación exclusivamente exploratoria.
    ranking.to_csv(resultados / 'ranking_exploratorio_fase3.csv', index=False)  # Exportamos la tabla para comparar con la Fase 2.
    espacial, detalle_espacial = comparar_municipios(entrenamiento)  # Medimos también la capacidad de extrapolación geográfica.
    espacial.to_csv(resultados / 'comparacion_municipios_fase3.csv', index=False)  # Guardamos RMSE, MAE y R² por municipios excluidos.
    detalle_espacial.to_csv(resultados / 'predicciones_municipios_fase3.csv', index=False)  # Conservamos los errores individuales de cada municipio.
    if args.rapido:  # Identificamos el modo abreviado solicitado por el equipo.
        detalle_figura = detalle.loc[(detalle['candidato'] == ranking.iloc[0]['candidato']) & (detalle['semilla'] == SEMILLAS[0])].copy()  # Seleccionamos un conjunto completo de predicciones fuera de muestra.
        resultado_anidado = None  # Indicamos que todavía no se efectuó la evaluación con selección interna.
    else:  # Ejecutamos la validación científica adicional en el modo completo.
        metricas_anidadas, detalle_figura, elecciones = seleccion_anidada(entrenamiento)  # Seleccionamos modelos dentro de cada fold sin consultar el conjunto externo.
        pd.DataFrame([metricas_anidadas]).to_csv(resultados / 'evaluacion_anidada_fase3.csv', index=False)  # Documentamos la estimación obtenida mediante la selección anidada.
        elecciones.to_csv(resultados / 'selecciones_internas_fase3.csv', index=False)  # Registramos el modelo elegido en cada una de las cinco pruebas.
        detalle_figura.to_csv(resultados / 'predicciones_anidadas_fase3.csv', index=False)  # Guardamos el detalle que origina las métricas principales.
        resultado_anidado = metricas_anidadas  # Conservamos las métricas para el informe y el archivo de configuración.
    errores_estados = []  # Preparamos una comparación regional de las predicciones externas.
    for estado, grupo in detalle_figura.groupby('estado'):  # Examinamos cada estado sin mezclar etiquetas ocultas.
        errores_estados.append({'estado': estado, 'parcelas': len(grupo), **calcular_metricas(grupo['real_t_ha'], grupo['predicho_t_ha'])})  # Medimos cuánto se equivoca el procedimiento en cada región.
    pd.DataFrame(errores_estados).to_csv(resultados / 'errores_por_estado_fase3.csv', index=False)  # Guardamos la evidencia de los posibles sesgos regionales.
    crear_graficas(comparacion, espacial, detalle_figura, resultados)  # Generamos figuras interpretables y reutilizables en el reporte.
    ganador = str(ranking.iloc[0]['candidato'])  # Identificamos un candidato meramente exploratorio con el menor error medio.
    borrador = exportar_borrador(entrenamiento, ocultas, ganador, resultados)  # Guardamos un borrador etiquetado explícitamente como no final.
    configuracion = {  # Documentamos las reglas, advertencias y versiones reales de esta ejecución.
        'estado': 'EXPERIMENTAL_NO_ENTREGAR',  # Impedimos confundir el experimento con una selección final para FIRA.
        'fecha_utc': datetime.now(timezone.utc).isoformat(),  # Registramos cuándo se ejecutó esta comparación.
        'python': platform.python_version(),  # Guardamos la versión de Python utilizada.
        'sklearn': sklearn.__version__,  # Guardamos la versión de scikit-learn.
        'xgboost': xgboost.__version__,  # Guardamos la versión real de XGBoost.
        'rapido': bool(args.rapido),  # Indicamos si utilizamos una o tres semillas.
        'semillas': list(SEMILLAS[:1] if args.rapido else SEMILLAS),  # Registramos las divisiones de evaluación.
        'parcelas_conocidas': len(entrenamiento),  # Comprobamos la cantidad de ejemplos supervisados.
        'parcelas_ocultas': len(ocultas),  # Comprobamos las predicciones pendientes del concurso.
        'candidatos': [candidato[0] for candidato in CANDIDATOS],  # Identificamos todos los modelos comparados.
        'mejor_exploratorio': ganador,  # Registramos el mínimo RMSE exploratorio, sin declarar que sea un ganador oficial.
        'cv_anidada': resultado_anidado,  # Guardamos métricas adicionales solamente cuando el experimento fue completo.
        'advertencias': ['La búsqueda previa de modelos limita la independencia de la evaluación anidada.', 'Los índices de 2025 incluyen observaciones durante el crecimiento; no prueban un pronóstico previo a sembrar.', 'La validación por municipios no equivale a la validación por parcelas.', 'Las 59 predicciones son experimentales y no contienen rendimientos reales de evaluación.'],  # Comunicamos las limitaciones antes de interpretar el resultado.
    }  # Cerramos la estructura de configuración reproducible.
    (resultados / 'configuracion_fase3.json').write_text(json.dumps(configuracion, ensure_ascii=False, indent=2), encoding='utf-8')  # Exportamos los metadatos en un formato fácil de inspeccionar.
    print('\nRANKING EXPLORATORIO\n' + ranking.to_string(index=False), flush=True)  # Mostramos el orden observado de los algoritmos.
    print('\nGENERALIZACIÓN ESPACIAL\n' + espacial.sort_values('rmse').to_string(index=False), flush=True)  # Mostramos la comparación por municipios desconocidos.
    print('\nEVALUACIÓN ANIDADA\n' + str(resultado_anidado), flush=True)  # Comunicamos si la selección interna se completó.
    print('\nRANGO 59 BORRADORES', float(borrador['RENDIMIENTO_PREDICHO_T_HA'].min()), float(borrador['RENDIMIENTO_PREDICHO_T_HA'].max()), flush=True)  # Revisamos que las estimaciones sean numéricamente plausibles.
    print('\nARCHIVOS GENERADOS:', resultados, flush=True)  # Facilitamos al equipo encontrar tablas y gráficas.
    print('ESTADO: SOLO EXPERIMENTAL, NO ENTREGAR A FIRA', flush=True)  # Reiteramos que aún debemos decidir y congelar el modelo.


# ==================== 14. INICIO PROTEGIDO DEL SCRIPT ====================
if __name__ == '__main__':  # Evitamos entrenar modelos simplemente por importar este módulo.
    main()  # Ejecutamos la Fase 3 cuando la invocamos expresamente.
