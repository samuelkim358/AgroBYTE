# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
# desarrollamos esta fase con una fila por parcela y sin utilizar los rendimientos ocultos de FIRA.
from pathlib import Path  # Construimos rutas válidas en cualquier computadora.
from rutas import carpeta_dataset, raiz_proyecto, validar_carpeta_salida  # Compartimos rutas y protección del dataset.
from datetime import datetime, timezone  # registramos la fecha UTC del experimento.
import json  # Importamos JSON para documentar los parámetros y resultados de cada experimento.
import argparse  # permitimos indicar rutas locales desde la terminal.
import warnings  # identificamos y controlamos avisos de las bibliotecas.
import numpy as np  # realizamos cálculos con matrices de valores numéricos.
import pandas as pd  # organizamos las parcelas y los resultados en tablas.
import joblib  # guardamos el modelo entrenado para usarlo en la aplicación.



# ==================== IMPORTACIÓN DE ALGORITMOS Y UTILIDADES DE MACHINE LEARNING ====================
from catboost import CatBoostRegressor  # probamos boosting con regularización para datos pequeños.
from sklearn.base import BaseEstimator, RegressorMixin  # integramos CatBoost con la interfaz de scikit-learn.
from sklearn.compose import ColumnTransformer  # aplicamos distintas transformaciones a números y categorías.
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor  # comparamos bosques y boosting de histogramas.
from sklearn.impute import SimpleImputer  # ajustamos valores faltantes solo con datos de entrenamiento.
from sklearn.linear_model import Ridge  # utilizamos una regresión lineal regularizada como referencia.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # medimos el error observado.
from sklearn.model_selection import StratifiedKFold, GroupKFold  # separamos parcelas y municipios sin cruzarlos.
from sklearn.pipeline import Pipeline  # encadenamos transformaciones y entrenamiento contra fugas.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # codificamos lugares y normalizamos números para Ridge.
import matplotlib  # preparamos gráficos reproducibles sin interfaz gráfica.
matplotlib.use('Agg')  # guardamos gráficos aunque ejecutemos Python en un servidor sin monitor.
import matplotlib.pyplot as plt  # visualizamos el rendimiento de los candidatos.



# ==================== 2. RUTAS Y CONFIGURACIÓN ====================
RAIZ_PROYECTO = raiz_proyecto()  # Localizamos automáticamente la raíz del proyecto desde este módulo.
BASE = carpeta_dataset()  # Localizamos los originales dentro de AGROBYTE PROJECT/DATASET_RETO_AGRO_2026.
PREV = RAIZ_PROYECTO / 'salidas' / 'fase1'  # reutilizamos las variables generadas en la fase 1 de este proyecto.
OUT = RAIZ_PROYECTO / 'salidas' / 'fase2'  # depositamos aquí las métricas y modelos nuevos de la fase 2.
MESES = tuple(range(4, 11))  # limitamos los índices al ciclo abril-octubre de 2025.
SEEDS = (42, 73, 2026)  # repetimos la validación para evitar depender de una única partición.

# separamos la lectura de los insumos y compruebo que el conjunto de evaluación permanece sin etiquetas.




# ==================== CARGA DE MATRIZ Y ETIQUETAS DE RENDIMIENTO ====================
def cargar_datos():  # definimos una función que une características y respuestas mediante el identificador.
    encontrados = sorted(BASE.glob('ID_area_rendimiento*.csv'))  # buscamos el CSV oficial con o sin sufijo como (2).
    if len(encontrados) != 1:  # evitamos elegir una fuente incorrecta cuando falta o está duplicada.
        raise FileNotFoundError(f'Necesito exactamente un CSV de rendimientos en {BASE}; encontré: {encontrados}')  # explicamos qué debe corregirse.
    objetivo = pd.read_csv(encontrados[0])  # leemos el CSV original de FIRA sin modificarlo.
    objetivo['ID_POLIGONO'] = objetivo['ID_POLIGONO'].str.strip()  # normalizamos identificadores sin modificar su significado.
    matriz = pd.read_csv(PREV / 'caracteristicas_nubes_20.csv')  # leemos las 166 características previamente generadas con nubes <= 20%.
    matriz['ID_POLIGONO'] = matriz['ID_POLIGONO'].str.strip()  # preparamos la llave única de relación.
    assert len(objetivo) == len(matriz) == 197  # exigimos exactamente 197 parcelas originales.
    assert matriz['ID_POLIGONO'].is_unique and objetivo['ID_POLIGONO'].is_unique  # impedimos duplicaciones por llave.
    assert set(matriz['ID_POLIGONO']) == set(objetivo['ID_POLIGONO'])  # exigimos cobertura geográfica completa.
    assert objetivo.loc[objetivo['CONJUNTO'].eq('PREDICCION'), 'RENDIMIENTO_T_HA'].isna().all()  # protegemos las 59 etiquetas ocultas.
    assert objetivo['RENDIMIENTO_T_HA'].notna().sum() == 138  # verificamos el total real de etiquetas conocidas.
    tabla = objetivo[['ID_POLIGONO', 'AREA_HA', 'CONJUNTO', 'RENDIMIENTO_T_HA']].merge(matriz.drop(columns=['AREA_HA']), on='ID_POLIGONO', validate='one_to_one')  # conservamos metadatos y variables sin duplicar el área.
    tabla = tabla.sort_values('ID_POLIGONO').reset_index(drop=True)  # ordenamos todas las parcelas por clave estable.
    entrenamiento = tabla.loc[tabla['CONJUNTO'].eq('ENTRENAMIENTO')].copy().reset_index(drop=True)  # aislamos los 138 ejemplos supervisados.
    prediccion = tabla.loc[tabla['CONJUNTO'].eq('PREDICCION')].copy().reset_index(drop=True)  # reservamos las 59 parcelas sin respuesta.
    return entrenamiento, prediccion  # devolvemos matrices independientes para evitar contaminación de evaluación.

# comparamos distintas familias de variables como un experimento controlado.




# ==================== SELECCIÓN CONTROLADA DE CARACTERÍSTICAS ====================
def columnas_para(matriz, grupo):  # construimos subconjuntos predefinidos sin consultar los valores de rendimiento.
    geografia = ['AREA_HA', 'latitud', 'longitud', 'estado', 'municipio']  # consideramos ubicación y superficie.
    todos = [col for col in matriz.columns if col not in ('ID_POLIGONO', 'RENDIMIENTO_T_HA', 'CONJUNTO')]  # excluimos ID, objetivo y bandera de evaluación.
    if grupo == 'geografia':  # detectamos el modelo exclusivamente geográfico.
        return geografia  # devolvemos cinco variables explícitas.
    if grupo == 'completo':  # detectamos el modelo con todas las características.
        return todos  # devolvemos las variables satelitales, climáticas, geográficas y topográficas.
    sat = [col for col in todos if col.startswith(('Sentinel_2_', 'Planet_', 'Landsat_'))]  # separamos índices de cada sensor.
    indices = ('ndvi_', 'evi_', 'fapar_', 'savi_', 'ndwi_', 'msi_', 'nddi_', 'msavi_', 'vi6t_')  # priorizamos vigor y humedad según una hipótesis agronómica previa.
    sat_reducidos = [col for col in sat if any(f'_{nombre}' in col for nombre in indices)]  # retiramos índices no seleccionados de forma no supervisada.
    if grupo == 'reducido':  # detectamos el experimento con subconjunto agronómico.
        otros = [col for col in todos if col not in sat and col not in geografia]  # conservamos temperatura, lluvia y terreno.
        return geografia + otros + sat_reducidos  # combinamos contexto territorial, clima e índices elegidos a priori.
    raise ValueError('Grupo de variables no reconocido: ' + grupo)  # detenemos el programa ante errores de configuración.

# adaptamos la interfaz de CatBoost para la versión instalada de scikit-learn sin modificar las predicciones.




# ==================== ADAPTADOR COMPATIBLE DE CATBOOST ====================
class CatBoostCompatible(RegressorMixin, BaseEstimator):  # implementomos las convenciones de estimador requeridas por el entorno.
    def __init__(self, iterations=350, depth=3, learning_rate=0.035, l2_leaf_reg=8, random_seed=42):  # expongomos hiperparámetros para clonación reproducible.
        self.iterations, self.depth, self.learning_rate = iterations, depth, learning_rate  # guardamos las opciones del boosting.
        self.l2_leaf_reg, self.random_seed = l2_leaf_reg, random_seed  # guardamos regularización y semilla.
    def fit(self, X, y):  # ajustamos una instancia de CatBoost dentro del fold vigente.
        self.estimator_ = CatBoostRegressor(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_seed, loss_function='RMSE', verbose=False, allow_writing_files=False, thread_count=2)  # construimos el modelo real.
        self.estimator_.fit(X, y)  # entrenamos con características ya procesadas y etiquetas permitidas.
        return self  # marcamos el adaptador como entrenado para el pipeline.
    def predict(self, X):  # envíomos los datos al modelo aprendido previamente.
        return self.estimator_.predict(X)  # devolvemos exactamente las predicciones del regresor CatBoost.

# transformamos categorías, ausencias y escalas usando únicamente cada conjunto de entrenamiento.




# ==================== CONSTRUCCIÓN DE PIPELINES Y ALGORITMOS ====================
def construir_modelo(nombre, X):  # creamos un pipeline nuevo para el candidato que se está evaluando.
    categorias = [c for c in ('estado', 'municipio') if c in X.columns]  # detectamos variables cualitativas.
    numericas = [c for c in X.columns if c not in categorias]  # separamos las variables continuas.
    pipe_num = [('imputar', SimpleImputer(strategy='median', keep_empty_features=True))]  # aprendomos medianas exclusivamente del fold.
    if nombre == 'ridge':  # normalizamos las variables para que Ridge penalice sus coeficientes de modo comparable.
        pipe_num.append(('escalar', StandardScaler()))  # estimamos centro y escala usando solo el entrenamiento.
    pasos = [('numericas', Pipeline(pipe_num), numericas)]  # preparamos las transformaciones numéricas.
    if categorias:  # agregamos la transformación categórica cuando hay región o municipio.
        pipe_cat = Pipeline([('imputar', SimpleImputer(strategy='most_frequent')), ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # evitamos fallos con municipios no vistos.
        pasos.append(('categoricas', pipe_cat, categorias))  # concatenomos categorías codificadas con valores numéricos.
    pre = ColumnTransformer(pasos, remainder='drop', sparse_threshold=0)  # garantizamos una matriz numérica consistente.
    if nombre == 'extratrees':  # implementomos un bosque extremadamente aleatorizado regularizado.
        regresor = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.85, random_state=42, n_jobs=1)  # reduzcomos sobreajuste con hojas no unitarias.
    elif nombre == 'catboost':  # probamos boosting de árboles poco profundos.
        regresor = CatBoostCompatible(iterations=350, depth=3, learning_rate=0.035, l2_leaf_reg=8, random_seed=42)  # usamos regularización fuerte dada la muestra pequeña.
    elif nombre == 'histboost':  # ensayomos boosting basado en histogramas.
        regresor = HistGradientBoostingRegressor(max_iter=170, learning_rate=0.045, max_leaf_nodes=7, min_samples_leaf=9, l2_regularization=5, random_state=42)  # limitamos la complejidad del árbol.
    elif nombre == 'ridge':  # probamos una relación lineal regularizada.
        regresor = Ridge(alpha=30.0)  # penalizomos coeficientes grandes para reducir varianza.
    else:  # prevengomos el uso de nombres de algoritmo desconocidos.
        raise ValueError('Algoritmo no reconocido: ' + nombre)  # interrumpomos una configuración inválida.
    return Pipeline([('preprocesamiento', pre), ('regresion', regresor)])  # devolvemos preprocesamiento y modelo unidos.

# describomos siete alternativas sin elegirlas por el rendimiento de las 59 parcelas ocultas.
CANDIDATOS = [('geo_extratrees', 'geografia', 'extratrees'), ('completo_extratrees', 'completo', 'extratrees'), ('reducido_extratrees', 'reducido', 'extratrees'), ('completo_catboost', 'completo', 'catboost'), ('reducido_catboost', 'reducido', 'catboost'), ('reducido_histboost', 'reducido', 'histboost'), ('reducido_ridge', 'reducido', 'ridge')]  # fijamos las hipótesis que vamos a contrastar.

# calculamos un indicador explícito de error usando exclusivamente predicciones fuera del entrenamiento del fold.




# ==================== CÁLCULO DE MAE, RMSE Y R² ====================
def metricas(y, pred):  # resumimos una colección de predicciones comparables.
    return {'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'mae': float(mean_absolute_error(y, pred)), 'r2': float(r2_score(y, pred))}  # medimos RMSE, MAE y R² en t/ha.

# comparamos candidatos en tres validaciones cruzadas para estudiar su sensibilidad a la selección de parcelas.




# ==================== VALIDACIÓN CRUZADA REPETIDA POR PARCELAS ====================
def comparacion_repetida(train):  # evaluamos las mismas 138 parcelas bajo tres semillas diferentes.
    resultados = []  # conservamos métricas agregadas por configuración y semilla.
    individuales = []  # guardamos cada predicción fuera de muestra para auditoría.
    y = train['RENDIMIENTO_T_HA'].to_numpy(dtype=float)  # extraemos únicamente las etiquetas visibles.
    for semilla in SEEDS:  # repetimos el procedimiento con distintos pliegues.
        particiones = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla).split(train, train['estado']))  # mantenemos presencia regional en cada pliegue.
        for etiqueta, grupo, algoritmo in CANDIDATOS:  # comparamos las siete familias predefinidas.
            X = train[columnas_para(train, grupo)]  # seleccionamos variables sin mirar rendimiento.
            pred = np.full(len(train), np.nan)  # reservamos un resultado para cada parcela etiquetada.
            for numero, (indices_train, indices_test) in enumerate(particiones, 1):  # separamos parcelas desconocidas del entrenamiento del fold.
                modelo = construir_modelo(algoritmo, X)  # creamos un estimador sin parámetros aprendidos antes.
                modelo.fit(X.iloc[indices_train], y[indices_train])  # ajustamos preprocesamiento y algoritmo sobre datos permitidos.
                valores = modelo.predict(X.iloc[indices_test])  # predigomos parcelas cuyo rendimiento ha quedado oculto temporalmente.
                pred[indices_test] = valores  # colocamos cada predicción en su parcela correspondiente.
                for fila, pronostico in zip(indices_test, valores):  # documentamos cada pronóstico individual de validación.
                    individuales.append({'semilla': semilla, 'candidato': etiqueta, 'fold': numero, 'ID_POLIGONO': train.iloc[fila]['ID_POLIGONO'], 'estado': train.iloc[fila]['estado'], 'municipio': train.iloc[fila]['municipio'], 'real_t_ha': float(y[fila]), 'predicho_t_ha': float(pronostico), 'error_t_ha': float(pronostico - y[fila])})  # permitimos reconstruir los errores sin reevaluar el modelo.
            assert np.isfinite(pred).all()  # verificamos que todas las parcelas tengan predicción OOF.
            resultados.append({'semilla': semilla, 'candidato': etiqueta, 'grupo': grupo, 'algoritmo': algoritmo, **metricas(y, pred)})  # registramos resultados por semilla sin escoger aún un ganador.
            print('CV', semilla, etiqueta, resultados[-1]['rmse'], flush=True)  # mostramos el avance y la métrica medida.
    return pd.DataFrame(resultados), pd.DataFrame(individuales)  # entregamos resumen y datos individuales auditables.

# sometomos la familia ganadora preliminar a una evaluación más exigente por municipios completos.




# ==================== VALIDACIÓN ESPACIAL POR MUNICIPIOS ====================
def validacion_municipios(train, nombres):  # impedimos que un mismo municipio participe simultáneamente en entrenamiento y validación.
    grupos = train['estado'].astype(str) + '/' + train['municipio'].astype(str)  # identificamos geográficamente cada agrupación.
    y = train['RENDIMIENTO_T_HA'].to_numpy(dtype=float)  # usamos los 138 rendimientos conocidos.
    particiones = list(GroupKFold(n_splits=5).split(train, groups=grupos))  # generamos cinco grupos de municipios independientes.
    filas = []  # registramos predicciones individuales por modelo.
    resumen = []  # almacenomos métricas por modelo para comparar generalización espacial.
    for etiqueta, grupo, algoritmo in CANDIDATOS:  # buscamos las configuraciones elegidas para el diagnóstico espacial.
        if etiqueta not in nombres:  # evitamos gastar cálculo en configuraciones no solicitadas.
            continue  # pasomos a la siguiente hipótesis.
        X = train[columnas_para(train, grupo)]  # seleccionamos sus mismas variables previamente definidas.
        pred = np.full(len(train), np.nan)  # preparamos un resultado fuera de municipio por parcela.
        for k, (i_tr, i_te) in enumerate(particiones, 1):  # separamos municipios completos del entrenamiento.
            modelo = construir_modelo(algoritmo, X)  # iniciamos un modelo sin aprendizaje previo.
            modelo.fit(X.iloc[i_tr], y[i_tr])  # ajustamos el algoritmo únicamente con municipios permitidos.
            pred[i_te] = modelo.predict(X.iloc[i_te])  # hagomos pronósticos para municipios no vistos.
            for i in i_te:  # documentamos una fila por predicción de municipio.
                filas.append({'candidato': etiqueta, 'fold': k, 'ID_POLIGONO': train.iloc[i]['ID_POLIGONO'], 'estado': train.iloc[i]['estado'], 'municipio': train.iloc[i]['municipio'], 'real_t_ha': float(y[i]), 'predicho_t_ha': float(pred[i]), 'error_t_ha': float(pred[i] - y[i])})  # identificamos en qué región falla cada algoritmo.
        assert np.isfinite(pred).all()  # garantizamos cobertura de todas las 138 etiquetas.
        resumen.append({'candidato': etiqueta, **metricas(y, pred)})  # medimos desempeño fuera de municipios conocidos.
        print('ESPACIAL', etiqueta, resumen[-1]['rmse'], flush=True)  # informamos el diagnóstico espacial medido.
    return pd.DataFrame(resumen), pd.DataFrame(filas)  # retornomos resultados resumidos y detallados.

# probamos la selección de modelos dentro de cada pliegue externo para limitar el optimismo de elegir entre modelos.




# ==================== VALIDACIÓN CRUZADA ANIDADA ====================
def evaluacion_anidada(train, orden):  # aplicamos CV externa de cinco folds e interna de tres.
    seleccionables = ('geo_extratrees', 'completo_extratrees', 'reducido_catboost')  # fijamos tres familias contrastantes sin elegirlas con el fold externo.
    alternativas = [c for c in CANDIDATOS if c[0] in seleccionables]  # recuperamos algoritmos y conjuntos de variables.
    y = train['RENDIMIENTO_T_HA'].to_numpy(dtype=float)  # mantenemos las etiquetas conocidas por identificador.
    exterior = StratifiedKFold(n_splits=5, shuffle=True, random_state=319).split(train, train['estado'])  # definimos pliegues externos distintos de las semillas exploratorias.
    filas = []  # guardamos el resultado externo de cada parcela.
    elegidos = []  # guardamos el modelo escogido internamente en cada fold.
    for no_fold, (i_train, i_test) in enumerate(exterior, 1):  # separamos prueba externa intocable del ajuste interno.
        interior_train = train.iloc[i_train].reset_index(drop=True)  # aislamos la porción disponible para búsqueda.
        interior_y = y[i_train]  # aislamos los rendimientos permitidos en búsqueda.
        cortes = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=100 + no_fold).split(interior_train, interior_train['estado']))  # construimos pliegues internos estratificados.
        scores = []  # acumulamos el error interno de cada alternativa.
        for etiqueta, grupo, algoritmo in alternativas:  # evaluamos alternativas sin observar la prueba externa.
            X_inner = interior_train[columnas_para(interior_train, grupo)]  # creamos características internas sin fuga del objetivo.
            pred_inner = np.full(len(interior_train), np.nan)  # preparamos predicciones internas independientes.
            for j_tr, j_test in cortes:  # repetimos el entrenamiento en los tres pliegues internos.
                mod = construir_modelo(algoritmo, X_inner)  # creamos una instancia fresca del estimador.
                mod.fit(X_inner.iloc[j_tr], interior_y[j_tr])  # ajustamos cada modelo con sus propios datos de entrenamiento.
                pred_inner[j_test] = mod.predict(X_inner.iloc[j_test])  # calculamos el error solo donde no entrené.
            scores.append((float(np.sqrt(mean_squared_error(interior_y, pred_inner))), etiqueta, grupo, algoritmo))  # cuantificamos el RMSE interno agregado.
        _, ganador, grupo, algoritmo = sorted(scores)[0]  # seleccionamos el RMSE interno más bajo sin consultar el fold externo.
        X_train = train.iloc[i_train][columnas_para(train, grupo)]  # preparamos variables de entrenamiento externo.
        X_test = train.iloc[i_test][columnas_para(train, grupo)]  # preparamos exactamente las mismas variables externas.
        modelo = construir_modelo(algoritmo, X_train)  # inicializamos un estimador de la familia seleccionada.
        modelo.fit(X_train, y[i_train])  # ajustamos de nuevo con todo el entrenamiento externo.
        pred_out = modelo.predict(X_test)  # estimamos las parcelas externas que nunca se usaron para escoger familia.
        elegidos.append({'fold_externo': no_fold, 'candidato': ganador, 'rmse_interno_mejor': sorted(scores)[0][0]})  # registramos la elección interna.
        for fila, predicho in zip(i_test, pred_out):  # documentamos las 138 predicciones externas independientes por pliegue.
            filas.append({'fold_externo': no_fold, 'candidato_elegido': ganador, 'ID_POLIGONO': train.iloc[fila]['ID_POLIGONO'], 'estado': train.iloc[fila]['estado'], 'municipio': train.iloc[fila]['municipio'], 'real_t_ha': float(y[fila]), 'predicho_t_ha': float(predicho), 'error_t_ha': float(predicho - y[fila])})  # preparamos diagnóstico detallado del error externo.
        print('ANIDADA', no_fold, ganador, flush=True)  # registramos la familia elegida en cada fold externo.
    oof = pd.DataFrame(filas).sort_values('ID_POLIGONO').reset_index(drop=True)  # reconstruyomos una predicción externa por parcela.
    assert len(oof) == 138 and oof['ID_POLIGONO'].is_unique  # validamos la integridad de la evaluación externa.
    return oof, pd.DataFrame(elegidos)  # devolvemos evidencia de la selección y de su desempeño externo.

# creamos una visualización clara para comparar familias sin ocultar que todo resultado sigue siendo experimental.




# ==================== VISUALIZACIÓN DE RESULTADOS ====================
def dibujar(scores, municipios):  # generamos una imagen de los modelos contrastados.
    medias = scores.groupby('candidato', as_index=False)['rmse'].agg(['mean', 'std']).reset_index().sort_values('mean')  # calculamos dispersión del RMSE entre semillas.
    fig, ax = plt.subplots(figsize=(11, 6))  # definimos un lienzo legible para el informe.
    ax.barh(medias['candidato'], medias['mean'], xerr=medias['std'], capsize=3)  # mostramos RMSE medio y variabilidad entre particiones.
    ax.invert_yaxis()  # colocamos el menor error en la parte superior.
    ax.set_xlabel('RMSE de predicción por parcela (t/ha)')  # expresamos las unidades del error.
    ax.set_title('AgroCebada FIRA 2026 — validación repetida (fase 2, preliminar)')  # identificamos la figura y su carácter exploratorio.
    fig.tight_layout()  # evitamos que los nombres de modelos queden cortados.
    fig.savefig(OUT / 'resultados' / 'rmse_repetido.png', dpi=160)  # guardamos la figura para revisión y documentación.
    plt.close(fig)  # liberomos memoria de visualización.

# ejecutamos el experimento completo, registro resultados y guardo un modelo marcado como NO FINAL.




# ==================== EJECUCIÓN PRINCIPAL DEL EXPERIMENTO ====================
def main():  # organizamos las pruebas y los entregables de forma reproducible.
    global BASE, PREV, OUT  # permitimos personalizar las rutas según la computadora donde se ejecuta el script.
    cli = argparse.ArgumentParser(description='Experimentos AgroCebada fase 2')  # construimos la interfaz de línea de comandos.
    cli.add_argument('--datos', type=Path, default=BASE)  # aceptamos la carpeta con el CSV oficial de rendimientos.
    cli.add_argument('--fase1', type=Path, default=PREV)  # aceptamos la carpeta de resultados de la fase 1.
    cli.add_argument('--salida', type=Path, default=OUT)  # aceptamos la ubicación donde guardar experimentos.
    cli.add_argument('--reusar-resultados', action='store_true')  # permitimos reutilizar resultados solo cuando el usuario lo solicita.
    args = cli.parse_args()  # leemos las opciones proporcionadas por el usuario.
    BASE, PREV, OUT = args.datos.resolve(), args.fase1.resolve(), args.salida.resolve()  # normalizamos las rutas sin modificar algoritmos.
    if OUT == BASE or BASE in OUT.parents:  # rechazamos salidas dentro de los archivos originales de FIRA.
        raise ValueError('No se permite guardar experimentos dentro de DATASET_RETO_AGRO_2026.')  # protegemos la carpeta fuente del concurso.
    (OUT / 'resultados').mkdir(parents=True, exist_ok=True)  # creamos el directorio de resultados.
    (OUT / 'modelos').mkdir(parents=True, exist_ok=True)  # creamos el directorio del modelo exportable.
    train, pred = cargar_datos()  # separamos entrenamiento conocido de evaluación oculta.
    assert train['RENDIMIENTO_T_HA'].notna().all() and pred['RENDIMIENTO_T_HA'].isna().all()  # reconfirmomos la frontera entre conjuntos.
    if args.reusar_resultados and (OUT / 'resultados' / 'resultados_cv_repetida.csv').exists():  # reutilizamos comparaciones solo si se solicitó expresamente.
        comparacion = pd.read_csv(OUT / 'resultados' / 'resultados_cv_repetida.csv')  # volvemos a leer las métricas ya producidas.
        detalle = pd.read_csv(OUT / 'resultados' / 'predicciones_cv_repetida.csv')  # recuperamos sus predicciones verificables.
    else:  # ejecutamos el experimento solamente cuando no existen resultados.
        comparacion, detalle = comparacion_repetida(train)  # comparamos siete candidatos en tres validaciones de cinco folds.
    comparacion.to_csv(OUT / 'resultados' / 'resultados_cv_repetida.csv', index=False)  # exportamos métricas por semilla y candidato.
    detalle.to_csv(OUT / 'resultados' / 'predicciones_cv_repetida.csv', index=False)  # conservamos evidencias de errores individuales.
    agregado = comparacion.groupby(['candidato', 'grupo', 'algoritmo'], as_index=False).agg(rmse_promedio=('rmse', 'mean'), rmse_desv=('rmse', 'std'), mae_promedio=('mae', 'mean'), r2_promedio=('r2', 'mean')).sort_values('rmse_promedio')  # resumimos la robustez ante distintas particiones.
    agregado.to_csv(OUT / 'resultados' / 'ranking_exploratorio.csv', index=False)  # guardamos el ranking provisional sin confundirlo con evaluación independiente.
    elegibles = agregado['candidato'].tolist()  # ordenamos los candidatos observados para diagnóstico posterior.
    if args.reusar_resultados and (OUT / 'resultados' / 'resultados_municipios.csv').exists():  # reutilizamos la prueba espacial solo con autorización explícita.
        espacios = pd.read_csv(OUT / 'resultados' / 'resultados_municipios.csv')  # recuperamos métricas registradas.
        detalle_espacial = pd.read_csv(OUT / 'resultados' / 'predicciones_municipios.csv')  # recuperamos resultados de cada parcela.
    else:  # ejecutamos la validación espacial si todavía no hay evidencia.
        espacios, detalle_espacial = validacion_municipios(train, list(dict.fromkeys(['geo_extratrees', 'completo_extratrees', 'reducido_catboost', 'completo_catboost', elegibles[0]])))  # verificamos modelos importantes en municipios no vistos.
    espacios.to_csv(OUT / 'resultados' / 'resultados_municipios.csv', index=False)  # entregamos las métricas espaciales.
    detalle_espacial.to_csv(OUT / 'resultados' / 'predicciones_municipios.csv', index=False)  # documentamos qué municipios tuvieron mayores errores.
    anidada, seleccion = evaluacion_anidada(train, elegibles)  # evaluamos la selección interna sin utilizar cada fold de prueba externo.
    anidada.to_csv(OUT / 'resultados' / 'predicciones_cv_anidada.csv', index=False)  # conservamos las predicciones anidadas externas.
    seleccion.to_csv(OUT / 'resultados' / 'modelos_elegidos_en_cv_anidada.csv', index=False)  # informamos qué modelo ganó cada elección interna.
    m_anidada = metricas(anidada['real_t_ha'], anidada['predicho_t_ha'])  # calculamos MAE, RMSE y R² de la estrategia de selección.
    m_anidada['protocolo'] = 'cv_anidada_por_parcelas'  # aclaromos qué representa este error y qué no representa.
    pd.DataFrame([m_anidada]).to_csv(OUT / 'resultados' / 'resumen_anidado.csv', index=False)  # dejomos listo el resultado metodológico principal.
    grupo_elegido = agregado.iloc[0]['grupo']  # identificamos el subconjunto exploratorio de menor RMSE promedio.
    algoritmo_elegido = agregado.iloc[0]['algoritmo']  # identificamos el algoritmo exploratorio que vamos a conservar.
    X_train = train[columnas_para(train, grupo_elegido)]  # preparamos las columnas finales conocidas sin etiquetas ocultas.
    X_pred = pred[columnas_para(pred, grupo_elegido)]  # preparamos las mismas columnas de las 59 parcelas ocultas.
    modelo = construir_modelo(algoritmo_elegido, X_train)  # instanciomos el modelo ganador exploratorio.
    modelo.fit(X_train, train['RENDIMIENTO_T_HA'].to_numpy(dtype=float))  # entrenamos con las 138 etiquetas disponibles.
    rendimiento = modelo.predict(X_pred)  # generamos 59 predicciones experimentales sin revisar su valor real.
    assert len(rendimiento) == 59 and np.isfinite(rendimiento).all()  # revisamos longitud y ausencia de valores inválidos.
    exportar = pred[['ID_POLIGONO', 'estado', 'municipio', 'AREA_HA']].copy()  # comenzamos la tabla de salida con datos de identificación.
    exportar['RENDIMIENTO_PREDICHO_T_HA'] = np.round(rendimiento, 5)  # registramos toneladas por hectárea, sin inventar etiquetas reales.
    exportar['PRODUCCION_ESTIMADA_T'] = np.round(rendimiento * pred['AREA_HA'].to_numpy(float), 5)  # convertimos rendimiento en producción por parcela.
    exportar.to_csv(OUT / 'resultados' / 'predicciones_59_EXPERIMENTALES_NO_ENTREGAR.csv', index=False)  # impedimos confundir el borrador con el archivo oficial.
    joblib.dump(modelo, OUT / 'modelos' / 'modelo_experimental_NO_FINAL.joblib')  # conservamos el pipeline ajustado junto al preprocesamiento.
    dibujar(comparacion, espacios)  # generamos una gráfica documentada del RMSE repetido.
    config = {'estado': 'EXPERIMENTAL_NO_FINAL', 'fecha_utc': datetime.now(timezone.utc).isoformat(), 'n_etiquetadas': len(train), 'n_prediccion': len(pred), 'n_features_completo': len(columnas_para(train, 'completo')), 'n_features_reducido': len(columnas_para(train, 'reducido')), 'n_features_geografia': len(columnas_para(train, 'geografia')), 'semillas_cv': list(SEEDS), 'candidatos': [x[0] for x in CANDIDATOS], 'mejor_modelo_cv_repetida': agregado.iloc[0]['candidato'], 'rmse_cv_repetida_promedio': float(agregado.iloc[0]['rmse_promedio']), 'evaluacion_anidada': m_anidada, 'avisos': ['La validación es interna; los datos oficiales ocultos nunca se usaron para métricas.', 'La familia de modelos se había explorado previamente; no equivale a test externo completamente independiente.', 'Los índices de abril-octubre 2025 producen una estimación retrospectiva, no un pronóstico antes de sembrar.', 'Las métricas por municipio evalúan extrapolación y pueden diferir radicalmente de la CV por parcelas.']}  # explicamos exactamente qué se comprobó y sus limitaciones.
    (OUT / 'resultados' / 'configuracion_fase2.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')  # guardamos metadatos en JSON legible.
    print('RANKING\n' + agregado.to_string(index=False), flush=True)  # mostramos la comparación real de modelos.
    print('ANIDADA\n' + json.dumps(m_anidada, indent=2), flush=True)  # mostramos la evaluación de selección interna.
    print('ESPACIAL\n' + espacios.to_string(index=False), flush=True)  # mostramos la robustez fuera de municipios.
    print(f'TERMINADO: {OUT}', flush=True)  # confirmamos la carpeta de salida creada.

if __name__ == '__main__':  # ejecutamos el experimento solo cuando inicio el archivo como programa.
    main()  # comenzamos la validación, el diagnóstico y la exportación provisional.
