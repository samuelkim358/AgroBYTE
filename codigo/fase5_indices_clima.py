"""Fase 5: estudiamos índices, clima y ablaciones sin tocar el modelo oficial."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # Habilitamos anotaciones claras para las funciones del módulo.
import argparse  # Recibimos rutas y parámetros desde la terminal.
import json  # Guardamos metadatos y controles reproducibles en JSON.
import re  # Identificamos familias de variables por el nombre de sus columnas.
from collections import defaultdict  # Reunimos nombres de columnas en grupos interpretables.
from datetime import datetime, timezone  # Registramos cuándo se realizó cada experimento.
from pathlib import Path  # Trabajamos con rutas independientes de Windows.
import numpy as np  # Calculamos permutaciones y errores estadísticos.
import pandas as pd  # Organizamos matrices de entrada y tablas de resultados.
import matplotlib  # Preparamos gráficas reproducibles sin depender de un escritorio.
matplotlib.use("Agg")  # Guardamos figuras en archivos sin abrir ventanas.
import matplotlib.pyplot as plt  # Creamos comparaciones visuales para el informe de FIRA.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Evaluamos predicciones en parcelas reservadas.
from sklearn.model_selection import StratifiedKFold  # Conservamos los estados en las particiones de validación.
from fase2_modelos_comparacion import construir_modelo  # Reutilizamos exactamente el mismo Extra Trees de Fase 2.
from rutas import raiz_proyecto  # Localizamos la raíz del proyecto sin rutas personales.


# ==================== 2. CONFIGURACIÓN Y CONTROLES DE DATOS ====================
RAIZ = raiz_proyecto()  # Identificamos la carpeta raíz de AgroBYTE.
SEEDS = (42, 73, 2026)  # Repetimos las mismas semillas de los experimentos anteriores.
SENSORES = ('Sentinel_2', 'Landsat', 'Planet')  # Distinguimos proveedores para no mezclar mediciones equivalentes sin control.
GEOGRAFIA = ('AREA_HA', 'latitud', 'longitud', 'estado', 'municipio')  # Fijamos las cinco entradas del modelo final.


def cargar_datos(matriz_path: Path, etiquetas_path: Path):  # Unimos características y rendimientos exclusivamente por ID de parcela.
    matriz = pd.read_csv(matriz_path)  # Leemos la matriz de Fase 1 al 20% de nubosidad, sin volver a generarla.
    etiquetas = pd.read_csv(etiquetas_path)  # Leemos los rendimientos oficiales y el indicador de conjunto.
    for tabla in (matriz, etiquetas):  # Revisamos las claves en ambas tablas.
        tabla['ID_POLIGONO'] = tabla['ID_POLIGONO'].astype(str).str.strip()  # Normalizamos espacios de los identificadores.
        if tabla['ID_POLIGONO'].duplicated().any():  # Detectamos parcelas duplicadas antes del cruce.
            raise ValueError('Identificadores duplicados en las fuentes FIRA.')  # Detenemos el análisis por riesgo de unión errónea.
    requeridas = {'ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA'}  # Declaramos el esquema mínimo de etiquetas.
    if not requeridas.issubset(etiquetas.columns):  # Comprobamos que exista la variable objetivo.
        raise ValueError('Faltan identificadores o etiquetas oficiales en el CSV.')  # Evitamos entrenar con columnas equivocadas.
    if len(matriz) != 197 or len(etiquetas) != 197:  # Comprobamos los totales originales del reto.
        raise ValueError('La matriz y las etiquetas deben incluir exactamente 197 parcelas.')  # Rechazamos matrices incompletas.
    if set(matriz.ID_POLIGONO) != set(etiquetas.ID_POLIGONO):  # Garantizamos correspondencia de parcelas.
        raise ValueError('Los identificadores de las características y etiquetas no coinciden.')  # Evitamos cruces parciales.
    tabla = etiquetas[['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA']].merge(matriz, on='ID_POLIGONO', validate='one_to_one')  # Unimos únicamente la respuesta necesaria.
    entrenamiento = tabla.loc[tabla.CONJUNTO.eq('ENTRENAMIENTO')].copy().sort_values('ID_POLIGONO').reset_index(drop=True)  # Aislamos rendimientos permitidos.
    ocultas = tabla.loc[tabla.CONJUNTO.eq('PREDICCION')].copy()  # Aislamos las 59 parcelas sin rendimiento.
    if len(entrenamiento) != 138 or len(ocultas) != 59:  # Verificamos que respetamos el reto 70/30.
        raise ValueError('Esperábamos 138 parcelas etiquetadas y 59 ocultas.')  # Detenemos cualquier cambio inesperado.
    if entrenamiento.RENDIMIENTO_T_HA.isna().any() or ocultas.RENDIMIENTO_T_HA.notna().any():  # Detectamos filtración de etiquetas.
        raise ValueError('Las etiquetas no respetan los conjuntos ENTRENAMIENTO y PREDICCION.')  # Impedimos el uso inadvertido de respuestas ocultas.
    return entrenamiento, ocultas  # Devolvemos los grupos separados para la auditoría.


# ==================== 3. CLASIFICACIÓN DE CARACTERÍSTICAS ====================
def familias_columnas(tabla):  # Separamos geografía, índices, controles de calidad, clima y topografía.
    columnas = [c for c in tabla.columns if c not in ('ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA')]  # Excluimos objetivo, identificador y conjunto.
    geo = [c for c in GEOGRAFIA if c in columnas]  # Conservamos las mismas cinco columnas del ganador.
    indices = [c for c in columnas if c.startswith(tuple(f'{s}_' for s in SENSORES)) and '_n_observaciones_' not in c and '_n_meses_' not in c]  # Identificamos índices reales.
    calidad = [c for c in columnas if c.startswith(tuple(f'{s}_' for s in SENSORES)) and c not in indices]  # Separamos recuentos de nubes y meses observados.
    clima = [c for c in columnas if c.startswith(('prec_mm_', 'tmin_c_', 'tmax_c_'))]  # Separamos precipitación y temperatura.
    topo = [c for c in ('elevacion_m', 'pendiente_grados') if c in columnas]  # Conservamos elevación y pendiente.
    usadas = set(geo + indices + calidad + clima + topo)  # Comprobamos que cada variable quede contabilizada.
    resto = [c for c in columnas if c not in usadas]  # Identificamos columnas desconocidas para no excluirlas silenciosamente.
    if len(geo) != 5 or not indices or not clima or resto:  # Evitamos ejecutar un análisis con un esquema inesperado.
        raise ValueError(f'Esquema inesperado: {len(geo)} geográficas, {len(indices)} índices, {len(clima)} clima. Sin clasificar: {resto}')  # Indicamos el problema exacto.
    return {'geo': geo, 'indices': indices, 'calidad': calidad, 'clima': clima, 'topo': topo}  # Conservamos familias auditables.


def configuraciones(f):  # Definimos hipótesis de comparación antes de examinar sus rendimientos.
    union = lambda *nombres: list(dict.fromkeys(c for nombre in nombres for c in f[nombre]))  # Combinamos familias sin repetir columnas.
    return {  # Formamos una ablación descriptiva sin alterar Extra Trees definitivo.
        'geo': union('geo'),  # Conservamos la línea base seleccionada.
        'solo_indices': union('indices'),  # Comprobamos cuánto explican solo los índices.
        'solo_clima': union('clima'),  # Comprobamos cuánto explica solamente el clima.
        'solo_topografia': union('topo'),  # Comprobamos cuánto explica solo la topografía.
        'geo_indices': union('geo', 'indices'),  # Calculamos la aportación incremental de los índices.
        'geo_clima': union('geo', 'clima'),  # Calculamos la aportación incremental del clima.
        'geo_topografia': union('geo', 'topo'),  # Calculamos la aportación incremental del relieve.
        'indices_clima': union('indices', 'clima'),  # Estudiamos información agronómica sin geografía directa.
        'geo_indices_clima': union('geo', 'indices', 'clima'),  # Comprobamos la combinación geografía-satélite-clima.
        'completo': union('geo', 'indices', 'calidad', 'clima', 'topo'),  # Reproducimos las características del modelo completo anterior.
        'geo_clima_historico': union('geo') + [c for c in f['clima'] if '2022_2024' in c],  # Evaluamos clima histórico sin utilizar el clima del ciclo 2025.
    }  # Entregamos grupos definidos sin mirar las etiquetas.


# ==================== 4. VALIDACIÓN CRUZADA DE ABLACIONES ====================
def particiones(train, semillas=SEEDS):  # Producimos las mismas divisiones para todos los modelos.
    for semilla in semillas:  # Repetimos los experimentos con semillas fijas.
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)  # Equilibramos la representación de estados entre folds.
        for fold, (it, iv) in enumerate(splitter.split(train, train['estado']), 1):  # Dejamos un subconjunto sin utilizar en entrenamiento.
            yield semilla, fold, it, iv  # Reutilizamos cada corte entre configuraciones.


def comparar_ablaciones(train, grupos, semillas=SEEDS):  # Contrastamos la contribución de cada familia en el mismo protocolo.
    y = train.RENDIMIENTO_T_HA.to_numpy(float)  # Conservamos solamente los 138 rendimientos conocidos.
    detalles = []  # Registramos el error de cada fold y sus metadatos.
    pronosticos = []  # Guardamos las predicciones fuera de muestra por parcela.
    for nombre, cols in grupos.items():  # Recorremos combinaciones acordadas de características.
        X = train[cols]  # Seleccionamos las variables sin acceder a la respuesta.
        for semilla, fold, it, iv in particiones(train, semillas):  # Usamos cortes idénticos entre configuraciones.
            modelo = construir_modelo('extratrees', X)  # Creamos el mismo algoritmo y preprocesamiento de Fase 2.
            modelo.fit(X.iloc[it], y[it])  # Ajustamos imputaciones y árboles solo en los datos del fold de entrenamiento.
            p = modelo.predict(X.iloc[iv])  # Estimamos parcelas reservadas.
            rmse = float(np.sqrt(mean_squared_error(y[iv], p)))  # Medimos error cuadrático por fold.
            detalles.append({'configuracion':nombre,'semilla':semilla,'fold':fold,'n_variables':len(cols),'n_val':len(iv),'rmse':rmse,'mae':float(mean_absolute_error(y[iv],p))})  # Registramos el mismo procedimiento para cada alternativa.
            for pos, pred in zip(iv, p):  # Conservamos predicciones individuales con IDs estables.
                pronosticos.append({'configuracion':nombre,'semilla':semilla,'fold':fold,'ID_POLIGONO':train.loc[pos,'ID_POLIGONO'],'estado':train.loc[pos,'estado'],'real_t_ha':float(y[pos]),'predicho_t_ha':float(pred)})  # Permiten contrastar resultados emparejados.
        print('ABLACION',nombre,'completada',flush=True)  # Comunicamos el progreso sin modificar el modelo oficial.
    oof = pd.DataFrame(pronosticos)  # Convertimos las predicciones en una tabla auditable.
    resumen = []  # Calculamos el RMSE para cada semilla sobre sus 138 predicciones fuera de muestra.
    for (nombre,semilla), g in oof.groupby(['configuracion','semilla']):  # Agrupamos sin mezclar semillas repetidas.
        resumen.append({'configuracion':nombre,'semilla':semilla,'rmse':float(np.sqrt(mean_squared_error(g.real_t_ha,g.predicho_t_ha))),'mae':float(mean_absolute_error(g.real_t_ha,g.predicho_t_ha)),'r2':float(r2_score(g.real_t_ha,g.predicho_t_ha))})  # Comparamos métricas con la misma definición de las fases anteriores.
    por_semilla = pd.DataFrame(resumen)  # Conservamos variación entre las tres semillas.
    ranking = por_semilla.groupby('configuracion').agg(rmse_promedio=('rmse','mean'),rmse_desv=('rmse','std'),mae_promedio=('mae','mean'),r2_promedio=('r2','mean')).reset_index().sort_values('rmse_promedio')  # Resumimos el rendimiento sin tratar folds como observaciones independientes.
    return ranking, por_semilla, pd.DataFrame(detalles), oof  # Proporcionamos resultados globales y auditables.


# ==================== 5. IMPORTANCIA DE PERMUTACIÓN AGRUPADA ====================
def grupos_de_indices(columnas):  # Agrupamos estadísticas de un mismo índice para evitar comparar meses aislados.
    by_index = defaultdict(list)  # Almacenamos NDVI, EVI y demás índices.
    by_sensor_index = defaultdict(list)  # Conservamos una segunda interpretación separada por sensor.
    for c in columnas:  # Analizamos nombres generados por nuestro pipeline satelital.
        patron = re.match(r'^(Sentinel_2|Landsat|Planet)_([a-z0-9]+)_promedio_(?:mes_\d{2}|ciclo_(?:mediana|maximo))$', c)  # Identificamos el índice por su proveedor y nombre.
        if patron is None:  # Evitamos confundir índices con metadatos de nubosidad.
            continue  # Continuamos con columnas relevantes únicamente.
        sensor, indice = patron.groups()  # Separamos sensor e índice de vegetación.
        by_index[indice.upper()].append(c)  # Agrupamos meses y sensores por familia de índice.
        by_sensor_index[f'{sensor}:{indice.upper()}'].append(c)  # Conservamos una segunda vista por proveedor.
    if not by_index:  # Comprobamos que identificamos correctamente los índices.
        raise ValueError('No se encontraron índices satelitales en las columnas del modelo.')  # Evitamos rankings vacíos.
    return dict(by_index), dict(by_sensor_index)  # Devolvemos dos agrupaciones complementarias.


def grupos_clima(columnas):  # Reunimos todos los meses y resúmenes de cada variable meteorológica.
    return {  # Respetamos tres magnitudes diferentes sin mezclarlas.
        'PRECIPITACION_CHIRPS': [c for c in columnas if c.startswith('prec_mm_')],  # Identificamos lluvia acumulada.
        'TEMPERATURA_MINIMA': [c for c in columnas if c.startswith('tmin_c_')],  # Identificamos temperatura mínima.
        'TEMPERATURA_MAXIMA': [c for c in columnas if c.startswith('tmax_c_')],  # Identificamos temperatura máxima.
    }  # Devolvemos los grupos de clima.


def importancia_agrupada(train, columnas, grupos, modelo_nombre, repeticiones=3):  # Evaluamos pérdida predictiva al romper una familia completa.
    X = train[columnas]  # Mantenemos solo las variables del modelo auxiliar.
    y = train.RENDIMIENTO_T_HA.to_numpy(float)  # Recuperamos etiquetas conocidas.
    resultados = []  # Acumulamos deltas de RMSE en parcelas reservadas.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Fijamos un único esquema consistente de cinco folds.
    for fold, (it, iv) in enumerate(splitter.split(train, train['estado']), 1):  # Repetimos únicamente en observaciones fuera del entrenamiento.
        modelo = construir_modelo('extratrees', X)  # Reutilizamos Extra Trees de Fase 2 sin tocar el candidato final.
        modelo.fit(X.iloc[it], y[it])  # Ajustamos imputaciones y árboles solo en entrenamiento.
        xv = X.iloc[iv].copy()  # Separamos el fold de evaluación.
        base_rmse = float(np.sqrt(mean_squared_error(y[iv],modelo.predict(xv))))  # Calculamos la línea base antes de perturbar información.
        for etiqueta, cols in grupos.items():  # Evaluamos índices o variables meteorológicas predeterminados.
            if not cols:  # Evitamos grupos sin columnas observadas.
                continue  # Seguimos con grupos válidos.
            for repeticion in range(repeticiones):  # Reducimos la dependencia de una sola permutación aleatoria.
                rng = np.random.default_rng(9000 + fold*101 + repeticion*17 + len(etiqueta))  # Generamos permutaciones reproducibles por grupo.
                permutacion = rng.permutation(len(xv))  # Desordenamos filas sin consultar sus etiquetas.
                xp = xv.copy()  # Evitamos contaminar otras variables y experimentos.
                xp.loc[:,cols] = xv.iloc[permutacion][cols].to_numpy()  # Permutamos el grupo conjuntamente y conservamos sus relaciones internas.
                perturbado = float(np.sqrt(mean_squared_error(y[iv],modelo.predict(xp))))  # Medimos error al retirar la asociación original de ese grupo.
                resultados.append({'modelo_auxiliar':modelo_nombre,'fold':fold,'repeticion':repeticion,'grupo':etiqueta,'columnas':len(cols),'rmse_original':base_rmse,'rmse_permutado':perturbado,'delta_rmse':perturbado-base_rmse})  # Registramos el aumento o disminución de error.
        print('PERMUTACION',modelo_nombre,'fold',fold,'completado',flush=True)  # Mostramos avance sin modificar modelos anteriores.
    detalle = pd.DataFrame(resultados)  # Guardamos todos los efectos de permutación.
    resumen = detalle.groupby(['modelo_auxiliar','grupo']).agg(n_columnas=('columnas','first'),delta_rmse_promedio=('delta_rmse','mean'),delta_rmse_mediana=('delta_rmse','median'),fraccion_delta_positivo=('delta_rmse',lambda v:float((v>0).mean()))).reset_index().sort_values('delta_rmse_promedio',ascending=False)  # Resumimos la importancia predictiva, no causal.
    return detalle, resumen  # Entregamos detalle y ranking descriptivo.


# ==================== 6. COMPARACIONES PAREADAS DE INCERTIDUMBRE ====================
def diferencias_pareadas(pred_oof, referencia='geo', semilla=42, remuestreos=2000):  # Estimamos la variabilidad de diferencias entre familias sobre las mismas parcelas.
    datos = pred_oof.loc[pred_oof.semilla.eq(semilla)].copy()  # Usamos una sola predicción fuera de muestra por parcela.
    base = datos.loc[datos.configuracion.eq(referencia),['ID_POLIGONO','real_t_ha','predicho_t_ha']].rename(columns={'predicho_t_ha':'referencia'})  # Recuperamos la línea base emparejada.
    rng = np.random.default_rng(2026)  # Fijamos la semilla del remuestreo para reproducibilidad.
    filas = []  # Guardamos intervalos descriptivos por comparación.
    for candidato in sorted(set(datos.configuracion)-{referencia}):  # Contrastamos cada familia con la misma referencia.
        rival = datos.loc[datos.configuracion.eq(candidato),['ID_POLIGONO','predicho_t_ha']].rename(columns={'predicho_t_ha':'alternativa'})  # Recuperamos predicciones auxiliares.
        unido = base.merge(rival,on='ID_POLIGONO',validate='one_to_one')  # Emparejamos parcelas idénticas en ambas estrategias.
        if len(unido)!=138:  # Impedimos comparar muestras no equivalentes.
            raise ValueError('No coinciden las 138 predicciones fuera de muestra.')  # Comunicamos una inconsistencia del experimento.
        real = unido.real_t_ha.to_numpy(float)  # Extraemos objetivos observados.
        ref = unido.referencia.to_numpy(float)  # Extraemos predicciones geográficas.
        alt = unido.alternativa.to_numpy(float)  # Extraemos predicciones de la comparación.
        dif = []  # Guardamos diferencias pareadas de RMSE de cada muestra bootstrap.
        for _ in range(remuestreos):  # Remuestreamos parcelas para estudiar sensibilidad descriptiva.
            idx = rng.integers(0,len(unido),len(unido))  # Seleccionamos parcelas con reemplazo sin volver a entrenar.
            dif.append(float(np.sqrt(np.mean((real[idx]-alt[idx])**2))-np.sqrt(np.mean((real[idx]-ref[idx])**2))))  # Medimos el RMSE alternativo menos el de referencia.
        val = float(np.sqrt(np.mean((real-alt)**2))-np.sqrt(np.mean((real-ref)**2)))  # Calculamos la diferencia observada sobre las 138 parcelas.
        filas.append({'alternativa':candidato,'referencia':referencia,'delta_rmse':val,'bootstrap_p025':float(np.quantile(dif,.025)),'bootstrap_p975':float(np.quantile(dif,.975)),'fraccion_delta_positivo':float((np.array(dif)>0).mean()),'advertencia':'Descriptivo: folds fueron usados en exploración; no es un test final independiente.'})  # Evitamos atribuir significación concluyente al resultado.
    return pd.DataFrame(filas).sort_values('delta_rmse')  # Devolvemos diferencias emparejadas auditables.


# ==================== 6. AUDITORÍA DE CALIDAD Y COBERTURA ====================
def resumen_calidad(train, ocultas, f):  # Describimos cuántos valores son válidos antes de imputación.
    filas = []  # Registramos calidad por familia y conjunto.
    for conjunto, tabla in [('ENTRENAMIENTO',train),('PREDICCION',ocultas)]:  # Comparamos cobertura entre regiones conocidas y ocultas.
        for grupo, cols in f.items():  # Analizamos cada bloque de variables.
            numericas = tabla[cols].apply(pd.to_numeric,errors='coerce') if grupo not in ('geo',) else tabla[[c for c in cols if c not in ('estado','municipio')]].apply(pd.to_numeric,errors='coerce')  # Excluimos categorías al cuantificar faltantes.
            filas.append({'conjunto':conjunto,'familia':grupo,'n_parcelas':len(tabla),'n_variables':len(cols),'n_valores_ausentes':int(numericas.isna().sum().sum()),'columnas_constantes':int(sum(numericas.nunique(dropna=True)<2))})  # Detectamos ausencia de observaciones y columnas sin variación.
    return pd.DataFrame(filas)  # Entregamos tabla de calidad separada de etiquetas.


# ==================== 7. GENERACIÓN DEL REPORTE ====================
def crear_reporte(out, ranking, imp_indices, imp_ind_geo, imp_clima, imp_clima_geo, calidad):  # Construimos conclusiones verificables con las cifras reales.
    def fila(nombre):  # Buscamos el resultado exacto de una configuración.
        sub = ranking.loc[ranking.configuracion.eq(nombre)]  # Evitamos recuperar accidentalmente otro experimento.
        return float(sub.iloc[0].rmse_promedio) if len(sub) else float('nan')  # Devolvemos el RMSE medido en validación.
    def top(tabla):  # Recuperamos el grupo con mayor contribución por permutación.
        if tabla.empty: return 'Sin grupos evaluables'  # No inventamos resultados si faltan variables.
        r = tabla.sort_values('delta_rmse_promedio',ascending=False).iloc[0]  # Elegimos la mayor pérdida descriptiva de error.
        return f"{r.grupo} (ΔRMSE medio={r.delta_rmse_promedio:+.4f} t/ha; {r.n_columnas} columnas)"  # Informamos magnitud y agrupación.
    diferencia = fila('completo') - fila('geo')  # Comparamos la combinación completa contra la línea base congelada.
    tabla_calidad = '\n'.join(['| Conjunto | Familia | Parcelas | Variables | Faltantes | Columnas constantes |', '|---|---|---:|---:|---:|---:|'] + [f"| {r.conjunto} | {r.familia} | {r.n_parcelas} | {r.n_variables} | {r.n_valores_ausentes} | {r.columnas_constantes} |" for r in calidad.itertuples(index=False)])  # Construimos una tabla legible sin instalar bibliotecas opcionales.
    texto = f'''# AgroBYTE · Fase 5 — Respuestas a las preguntas guía de FIRA\n\n**Estado:** análisis explicativo AUXILIAR. No cambia el modelo candidato Extra Trees geográfico ni sus 59 predicciones.\n\n## Datos y protocolo\n- Datos oficiales: 197 parcelas (138 con etiqueta, 59 ocultas).\n- Características de Fase 1: abril–octubre 2025, satélites filtrados a nubosidad ≤20%, clima CHIRPS y CHIRTS-ERA5, relieve INEGI.\n- Comparación de características: ExtraTreesRegressor de Fase 2, validación estratificada por estado de 5 pliegues × semillas 42, 73, 2026.\n- Importancia de permutación **agrupada**: 5 pliegues con semilla 42, tres permutaciones por grupo; ΔRMSE positivo implica que la variable resultó útil para ese modelo auxiliar.\n- Transformaciones e imputación aprendidas dentro de cada fold. Sin entrenar con etiquetas de las 59 parcelas.\n\n## Pregunta 1: ¿Qué índice satelital es más importante?\n- **Modelo con solo índices:** {top(imp_indices)}.\n- **Modelo geografía + índices:** {top(imp_ind_geo)}.\n- Interpretación: rankings de dependencia predictiva dentro del modelo indicado; sensores correlacionados, no importancia causal. Ver tablas por sensor y columna.\n\n## Pregunta 2: ¿Qué variable climática tiene mayor peso?\n- **Modelo solo clima:** {top(imp_clima)}.\n- **Modelo geografía + clima:** {top(imp_clima_geo)}.\n- Los grupos incluyen meses de 2025 y promedios históricos 2022–24.\n\n## Pregunta 3: ¿Qué combinación de variables funciona mejor?\n- Línea base geográfica: RMSE CV={fila('geo'):.4f} t/ha.\n- Geografía + índices: RMSE CV={fila('geo_indices'):.4f} t/ha.\n- Geografía + clima: RMSE CV={fila('geo_clima'):.4f} t/ha.\n- Todas las variables: RMSE CV={fila('completo'):.4f} t/ha (diferencia respecto a geografía: {diferencia:+.4f} t/ha).\n- **Precaución:** el modelo principal se seleccionó antes de esta exploración. La comparación auxiliar no demuestra que el mínimo RMSE encontrado ahora se generalice a nuevas parcelas, pues no repetimos selección anidada para estas combinaciones.\n\n## Límites científicos\n1. La importancia agrupada depende de la familia de variables usada para entrenar el modelo auxiliar; no es un efecto causal de los índices ni una propiedad universal de la cebada.\n2. Medianas y máximos de índices de abril–octubre de 2025 y clima del mismo periodo podrían incluir información posterior a la siembra/cosecha. Son válidos para un estudio retrospectivo, **no prueban predicción antes de sembrar**.\n3. La nubosidad se filtró al 20% por una decisión de ingeniería previa; no se ha demostrado que ese umbral sea óptimo para todas las variables.\n4. El conjunto de solo 138 etiquetas y la correlación satelital elevan la variabilidad de rankings. Folds y permutaciones repetidas no son muestras independientes.\n5. Nuestros mapas de rasters climáticos muestrean el centroide de parcela; distintos terrenos pueden compartir píxel climático, sin que esto demuestre causalidad.\n6. Las 59 etiquetas reales no están disponibles, por lo que su error no puede comprobarse.\n\n## Calidad de datos\n{tabla_calidad}\n\n## Resultado de la decisión\nMantenemos congelado el Extra Trees geográfico para las 59 parcelas de FIRA. Presentamos esta Fase 5 como evidencia complementaria para contestar las preguntas guía. Si encontráramos una configuración nueva aparentemente superior, requeriría su propia validación anidada antes de sustituir el modelo principal.\n'''
    path = out/'INFORME_FASE5.md'  # Definimos la ruta del documento explicativo.
    path.write_text(texto,encoding='utf-8')  # Exportamos el informe para revisarlo con el equipo.
    return path  # Entregamos la ubicación exacta del informe.


# ==================== 8. GRÁFICAS PARA COMUNICACIÓN CIENTÍFICA ====================
def guardar_graficas(out, ranking, indices, indices_geo, clima, clima_geo):  # Exportamos comparaciones legibles para el jurado.
    graficas = [  # Seleccionamos salidas verificables y no figuras puramente decorativas.
        ('01_combinaciones_rmse.png',ranking.sort_values('rmse_promedio',ascending=False),'configuracion','rmse_promedio','RMSE por conjunto de variables','RMSE (t/ha)'),  # Ilustramos la prueba de ablación.
        ('02_indices_solo_satelite.png',indices.sort_values('delta_rmse_promedio'),'grupo','delta_rmse_promedio','Importancia agrupada: solo índices','Δ RMSE al permutar (t/ha)'),  # Mostramos índices sin geografía.
        ('03_indices_con_geografia.png',indices_geo.sort_values('delta_rmse_promedio'),'grupo','delta_rmse_promedio','Importancia de índices al incluir geografía','Δ RMSE al permutar (t/ha)'),  # Mostramos cambios de importancia al controlar ubicación.
        ('04_clima_solo.png',clima.sort_values('delta_rmse_promedio'),'grupo','delta_rmse_promedio','Importancia agrupada: clima','Δ RMSE al permutar (t/ha)'),  # Contrastamos lluvia y temperaturas.
        ('05_clima_con_geografia.png',clima_geo.sort_values('delta_rmse_promedio'),'grupo','delta_rmse_promedio','Importancia de clima al incluir geografía','Δ RMSE al permutar (t/ha)'),  # Contrastamos información meteorológica incremental.
    ]  # Cerramos la lista de gráficas.
    for nombre, datos, col_etiqueta, col_valor, titulo, eje in graficas:  # Creamos archivos individuales para cada pregunta guía.
        fig,ax=plt.subplots(figsize=(9, max(3.7, .49*len(datos)+1.4)))  # Ajustamos tamaño al número de variables.
        ax.barh(datos[col_etiqueta].astype(str),datos[col_valor].astype(float))  # Dibujamos métricas observadas sin inventar escalas.
        ax.axvline(0,color='black',lw=.8) if col_valor=='delta_rmse_promedio' else None  # Distinguimos contribuciones negativas y positivas.
        ax.set_xlabel(eje)  # Indicamos magnitudes y unidades.
        ax.set_title(titulo)  # Describimos qué experimento representa la figura.
        ax.grid(axis='x',alpha=.18)  # Facilitamos la lectura sin distorsionar valores.
        fig.tight_layout()  # Evitamos títulos o etiquetas recortados.
        fig.savefig(out/nombre,dpi=145,bbox_inches='tight')  # Exportamos una figura por análisis para el informe.
        plt.close(fig)  # Liberamos memoria del programa tras cada gráfica.


# ==================== 8. PROGRAMA PRINCIPAL Y EXPORTACIONES ====================
def main():  # Ejecutamos la investigación sin sobrescribir ninguna fase anterior.
    cli = argparse.ArgumentParser(description='Fase 5: índices satelitales, clima y combinaciones para FIRA')  # Explicamos el propósito al equipo.
    cli.add_argument('--matriz',type=Path,default=RAIZ/'salidas'/'fase1'/'caracteristicas_nubes_20.csv')  # Recuperamos características ya creadas.
    cli.add_argument('--etiquetas',type=Path,default=None)  # Permitimos identificar un CSV oficial con sufijos de Windows.
    cli.add_argument('--salida',type=Path,default=RAIZ/'salidas'/'fase5')  # Escribimos exclusivamente en la nueva fase.
    args = cli.parse_args()  # Interpretamos las rutas recibidas.
    if args.etiquetas is None:  # Buscamos el CSV de rendimientos que el equipo conserva en su carpeta de datos.
        encontrados = sorted((RAIZ/'DATASET_RETO_AGRO_2026').glob('ID_area_rendimiento*.csv'))  # Admitimos sufijos como (2) de Windows.
        if len(encontrados) != 1:  # Rechazamos fuentes ambiguas o ausentes.
            raise FileNotFoundError(f'Esperábamos un solo CSV de rendimientos. Encontramos: {encontrados}')  # Informamos el problema de nombres.
        args.etiquetas = encontrados[0]  # Elegimos la única fuente oficial encontrada.
    args.salida.mkdir(parents=True,exist_ok=True)  # Creamos una carpeta aislada para los resultados auxiliares.
    train, ocultas = cargar_datos(args.matriz,args.etiquetas)  # Verificamos identidad y etiquetas antes de entrenar.
    f = familias_columnas(train)  # Separamos los cinco grupos de información disponibles.
    comb = configuraciones(f)  # Definimos los experimentos a comparar de forma transparente.
    calidad = resumen_calidad(train,ocultas,f)  # Verificamos faltantes y constantes por familia.
    ranking, semillas, folds, pred_oof = comparar_ablaciones(train,comb)  # Realizamos ablaciones con las mismas validaciones.
    bootstrap = diferencias_pareadas(pred_oof)  # Revisamos la incertidumbre descriptiva en comparaciones con geografía.
    idx, por_sensor = grupos_de_indices(f['indices'])  # Clasificamos familias de índices y fuentes satelitales.
    imp_i_det, imp_i = importancia_agrupada(train,f['indices'],idx,'solo_indices')  # Medimos índices sin información geográfica.
    imp_gi_det, imp_gi = importancia_agrupada(train,comb['geo_indices'],idx,'geo_indices')  # Medimos índices al controlar geografía.
    imp_c_det, imp_c = importancia_agrupada(train,f['clima'],grupos_clima(f['clima']),'solo_clima')  # Medimos clima sin coordenadas.
    imp_gc_det, imp_gc = importancia_agrupada(train,comb['geo_clima'],grupos_clima(f['clima']),'geo_clima')  # Medimos clima después de la geografía.
    imp_si_det, imp_si = importancia_agrupada(train,f['indices'],por_sensor,'indices_por_sensor')  # Evaluamos diferencias por proveedor sin fusionarlos.
    for nombre, tabla in [('ranking_ablaciones.csv',ranking),('metricas_por_semilla.csv',semillas),('metricas_por_fold.csv',folds),('predicciones_fuera_de_muestra.csv',pred_oof),('calidad_caracteristicas.csv',calidad),('diferencias_bootstrap_pareadas.csv',bootstrap),('importancia_indices.csv',imp_i),('importancia_indices_con_geografia.csv',imp_gi),('importancia_clima.csv',imp_c),('importancia_clima_con_geografia.csv',imp_gc),('importancia_indices_por_sensor.csv',imp_si),('permutaciones_detalle.csv',pd.concat([imp_i_det,imp_gi_det,imp_c_det,imp_gc_det,imp_si_det],ignore_index=True))]:  # Enumeramos salidas que ayudan al jurado a reproducir el análisis.
        tabla.to_csv(args.salida/nombre,index=False)  # Guardamos cada tabla sin modificar originales ni fases 1–4.
    meta = {'generado_utc':datetime.now(timezone.utc).isoformat(),'entrenamiento':len(train),'prediccion_oculta':len(ocultas),'semillas':list(SEEDS),'folds':5,'nubosidad_max_pct':20,'modelo':'ExtraTrees Fase2 n_estimators=150 min_samples_leaf=3 max_features=0.85','n_variables_matriz':sum(len(v) for v in f.values()),'familias':{k:len(v) for k,v in f.items()},'nota':'SOLO EXPLICATIVO: ninguna predicción del modelo Fase 4 ha sido recalculada'}  # Documentamos parámetros y su carácter auxiliar.
    (args.salida/'protocolo.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')  # Registramos la trazabilidad para FIRA.
    reporte = crear_reporte(args.salida,ranking,imp_i,imp_gi,imp_c,imp_gc,calidad)  # Redactamos las conclusiones basadas en resultados medidos.
    print('\nRANKING DE COMBINACIONES\n',ranking.to_string(index=False),flush=True)  # Mostramos comparaciones entre familias.
    print('\nINDICES SOLO SATELITES\n',imp_i.head(12).to_string(index=False),flush=True)  # Mostramos ranking de índices sin geografía.
    print('\nINDICES CON GEOGRAFIA\n',imp_gi.head(12).to_string(index=False),flush=True)  # Mostramos cuánto aporta la información satelital adicional.
    print('\nCLIMA SOLO\n',imp_c.to_string(index=False),flush=True)  # Mostramos el ranking climático sin geografía.
    print('\nCLIMA CON GEOGRAFIA\n',imp_gc.to_string(index=False),flush=True)  # Mostramos la dependencia climática adicional.
    print('\nDIFERENCIAS PAREADAS RESPECTO A GEOGRAFIA\n',bootstrap[['alternativa','delta_rmse','bootstrap_p025','bootstrap_p975']].to_string(index=False),flush=True)  # Documentamos sensibilidad de diferencias.
    print('\nINFORME:',reporte,flush=True)  # Indicamos la ruta de resultados verificables.


if __name__ == '__main__':  # Evitamos iniciar entrenamientos al importar este módulo desde pruebas.
    main()  # Iniciamos la Fase 5 únicamente cuando el usuario la solicita.
