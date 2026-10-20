"""AgroBYTE — procesamiento y modelos iniciales sin usar rendimientos ocultos.

Ejecucion: python codigo/experimento_agrocebada.py --datos /ruta/archivos_FIRA --salida resultados
Requiere: pandas numpy geopandas rasterio pyproj shapely scikit-learn joblib
Esta version hace una comparacion exploratoria, no una evaluacion externa independiente.
"""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
from __future__ import annotations  # habilitamos anotaciones modernas de tipos para usar versiones actuales de Python.
import json  # Importamos JSON para documentar los parámetros y resultados de cada experimento.
import argparse  # importamos esta biblioteca para apoyar el procesamiento reproducible.
import re  # importamos esta biblioteca para apoyar el procesamiento reproducible.
import zipfile  # importamos esta biblioteca para apoyar el procesamiento reproducible.
from pathlib import Path  # Construimos rutas portables sin unir cadenas manualmente.
from rutas import carpeta_dataset, raiz_proyecto, validar_carpeta_salida  # Compartimos las ubicaciones y la protección de los datos.
from datetime import datetime, timezone  # registramos en UTC el momento de generación de cada experimento.
from collections import defaultdict  # agrupamos mediciones del mismo tipo antes de resumirlas.
import numpy as np  # realizamos operaciones matemáticas con vectores y matrices.
import pandas as pd  # leemos CSV y organizamos los datos en tablas.
import geopandas as gpd  # leemos parcelas y cálculos espaciales en formatos geográficos.
import rasterio  # abrimos archivos ráster con lluvia, temperatura y terreno.
from rasterio.io import MemoryFile  # leemos un ráster comprimido directamente en memoria.
from pyproj import Transformer  # transformamos coordenadas antes de extraer valores de los mapas.



# ==================== IMPORTACIÓN DE MODELOS Y VALIDACIÓN DE MACHINE LEARNING ====================
from sklearn.ensemble import ExtraTreesRegressor  # preparamos el modelo de árboles extremadamente aleatorizados.
from sklearn.impute import SimpleImputer  # rellenamos datos ausentes con estadísticas aprendidas del entrenamiento.
from sklearn.compose import ColumnTransformer  # procesamos variables numéricas y categóricas por separado.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # convertimos categorías a números y escalamos cuando el algoritmo lo requiere.
from sklearn.pipeline import Pipeline  # agrupamos transformaciones y regresión en una cadena reproducible.
from sklearn.linear_model import Ridge  # comparamos una regresión lineal regularizada.
from sklearn.model_selection import StratifiedKFold, GroupKFold  # dividimos parcelas en pliegues preservando estados o separando municipios.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # calculamos MAE, RMSE y R² para comparar estimaciones.
from sklearn.base import clone  # creamos nuevas copias no entrenadas de nuestros modelos.
import joblib  # serializamos modelos para usos posteriores, si corresponde.

# encontramos la carpeta principal del proyecto a partir de este archivo, incluso si abrimos la terminal en otra ubicación.



# ==================== 2. RUTAS Y CONFIGURACIÓN ====================
RAIZ_PROYECTO = raiz_proyecto()  # Localizamos la carpeta principal sin depender de una ruta de Windows.
# leemos la ruta de datos indicada por el equipo, sin copiar ni modificar sus originales.
CARPETA_DATOS = carpeta_dataset()  # Buscamos los archivos oficiales en DATASET_RETO_AGRO_2026, junto al código.
# separamos físicamente las salidas de la primera fase de la carpeta original de FIRA.
CARPETA_SALIDA = RAIZ_PROYECTO / 'salidas' / 'fase1'  # guardamos resultados nuevos exclusivamente dentro del proyecto.

SEED=42  # fijamos una semilla para repetir divisiones y elecciones aleatorias del experimento.
MESES=list(range(4,11))  # seleccionamos abril a octubre, la temporada de estudio del rendimiento 2025.
SENSORS={'Sentinel-2': ['ndvi_promedio','evi_promedio','fapar_promedio','savi_promedio','ndwi_promedio','lai_promedio','msi_promedio','nddi_promedio','ndsvi_promedio'],  # definimos qué índices satelitales revisaremos por cada proveedor.
         'Landsat':['vi6t_promedio'],  # seleccionamos VI6T como característica espectral-térmica de Landsat.
         'Planet':['ndvi_promedio','evi_promedio','lai_promedio','msavi_promedio']}  # seleccionamos los cuatro índices que provee Planet.




# ==================== BÚSQUEDA DE ARCHIVOS POR NOMBRE ====================
def find_one(base,pattern):  # iniciamos la búsqueda de un archivo FIRA usando su patrón de nombre.
    found=list(base.glob(pattern))  # buscamos coincidencias del nombre requerido dentro de la carpeta de datos.
    if len(found) != 1:  # comprobamos la condición antes de continuar con el procesamiento.
        raise FileNotFoundError(  # detenemos el proceso con un mensaje explicativo cuando los datos no cumplen el requisito.
            f"No encontramos exactamente un archivo con patrón {pattern!r} en: {base.resolve()}\n"  # preparamos un mensaje que indica qué nombre de archivo buscamos.
            f"Archivos que coincidieron: {found}\n"  # mostramos las coincidencias reales para depurar nombres duplicados.
            "Solución: revisamos que DATASET_RETO_AGRO_2026 esté dentro del proyecto o utilizamos "  # explicamos dónde cambiar la ruta si falta un CSV.
            "--datos con la ruta de la carpeta que contiene los archivos originales."  # informamos la alternativa de indicar la ruta mediante --datos.
        )  # cerramos correctamente la expresión iniciada en líneas anteriores.
    return found[0]  # devolvemos el resultado calculado por esta función.




# ==================== VERIFICACIÓN DE INSUMOS OFICIALES ====================
def verificar_insumos(base):  # verificamos que estén todos los insumos antes del procesamiento.
    # comprobamos primero que la carpeta que contiene los datos originales realmente existe.
    if not base.is_dir():  # comprobamos la condición antes de continuar con el procesamiento.
        raise FileNotFoundError(  # detenemos el proceso con un mensaje explicativo cuando los datos no cumplen el requisito.
            f"La carpeta de datos no existe: {base.resolve()}\n"  # señalamos la ruta inexistente para poder corregirla.
            "Revisamos que DATASET_RETO_AGRO_2026 esté dentro del proyecto o usamos "  # explicamos cómo corregir el archivo de configuración.
            "--datos para indicar otra ubicación del dataset."  # ofrecemos una ruta alternativa sin mover ni copiar archivos.
        )  # cerramos correctamente la expresión iniciada en líneas anteriores.
    # declaramos las ocho piezas que necesitamos para ejecutar la lectura completa de los archivos FIRA.
    necesarios = [  # enumeramos las entradas mínimas antes de iniciar cálculos costosos.
        ('Rendimientos y áreas', 'ID_area_rendimiento*.csv'),  # comprobamos el CSV de rendimientos reales y superficies.
        ('Satélite Básico', 'Conjunto_datos_BASICO*.csv'),  # comprobamos el CSV de Sentinel-2 y Landsat.
        ('Satélite PRO', 'Conjunto_datos_PRO*.csv'),  # comprobamos el CSV de Planet.
        ('Polígonos SHP', 'Parcelas_Reto_AGC_CONJUNTO.shp'),  # comprobamos los límites de las 197 parcelas.
        ('Índice SHX', 'Parcelas_Reto_AGC_CONJUNTO.shx'),  # comprobamos el índice geométrico que acompaña al SHP.
        ('Atributos DBF', 'Parcelas_Reto_AGC_CONJUNTO.dbf'),  # comprobamos las columnas de atributos del Shapefile.
        ('Sistema PRJ', 'Parcelas_Reto_AGC_CONJUNTO.prj'),  # comprobamos la definición del sistema de coordenadas.
        ('Precipitación', 'Reto_AgroCebada_CHIRPS*.zip'),  # comprobamos el ZIP con lluvia mensual CHIRPS.
        ('Temperatura', 'Reto_AgroCebada_Temperatura*.zip'),  # comprobamos el ZIP con temperaturas mensuales.
        ('Topografía', 'Reto_AgroCebada_Topografia*.zip'),  # comprobamos el ZIP con altitud y pendiente.
    ]  # cerramos la estructura de datos iniciada en las líneas anteriores.
    # informamos exactamente qué carpeta vamos a revisar, para que se pueda identificar un error de ruta.
    print('Carpeta de datos:', base.resolve(), flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    faltantes = []  # iniciamos la lista de insumos que no encontramos o están duplicados.
    for descripcion, patron in necesarios:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        encontrados = list(base.glob(patron))  # encontramos archivos cuyos nombres corresponden al patrón solicitado.
        if len(encontrados) == 1:  # comprobamos la condición antes de continuar con el procesamiento.
            print(f'  OK: {descripcion} -> {encontrados[0].name}', flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
        else:  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
            print(f'  FALTA O DUPLICADO: {descripcion} | patrón: {patron} | coincidencias: {len(encontrados)}', flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
            faltantes.append(patron)  # registramos el nombre que debemos revisar antes de entrenar.
    if faltantes:  # comprobamos la condición antes de continuar con el procesamiento.
        raise FileNotFoundError(  # detenemos el proceso con un mensaje explicativo cuando los datos no cumplen el requisito.
            'Revisa la carpeta anterior. Falta alguno de estos archivos o hay duplicados: '  # explicamos si un archivo falta o está duplicado.
            + ', '.join(faltantes)  # mostramos todos los patrones que fallaron.
        )  # cerramos correctamente la expresión iniciada en líneas anteriores.
    print('Todos los insumos esenciales están presentes. Aún no he entrenado el modelo.', flush=True)  # mostramos en la terminal el avance o el resultado de la operación.




# ==================== LECTURA DE GEOMETRÍAS Y RENDIMIENTOS ====================
def read_geodata(base):  # reunimos geometrías y rendimientos sin perder la identidad de ninguna parcela.
    data=pd.read_csv(find_one(base,'ID_area_rendimiento*.csv')).drop(columns=['Unnamed: 4'], errors='ignore')  # leemos la tabla de 197 parcelas y sus rendimientos conocidos u ocultos.
    data['ID_POLIGONO']=data['ID_POLIGONO'].astype(str).str.strip()  # normalizamos el identificador para poder unir correctamente los archivos.
    assert data.ID_POLIGONO.nunique()==197, 'ID_POLIGONO debe ser unico'  # comprobamos una condición de integridad para detectar errores de datos.
    assert data['RENDIMIENTO_T_HA'].notna().sum()==138  # comprobamos una condición de integridad para detectar errores de datos.
    assert data.loc[data.CONJUNTO.eq('PREDICCION'),'RENDIMIENTO_T_HA'].isna().all()  # comprobamos una condición de integridad para detectar errores de datos.
    shp=gpd.read_file(find_one(base,'Parcelas_Reto_AGC_CONJUNTO.shp'))  # leemos el polígono geográfico original de cada parcela.
    shp=shp.rename(columns={'ID_POLIGON':'ID_POLIGONO','Estado':'estado','Municipio':'municipio','área_ha':'area_shp_ha'})  # unificamos nombres de columnas para trabajar con una misma nomenclatura.
    shp['ID_POLIGONO']=shp.ID_POLIGONO.astype(str).str.strip()  # normalizamos los identificadores geográficos.
    assert shp.ID_POLIGONO.is_unique and len(shp)==197  # comprobamos una condición de integridad para detectar errores de datos.
    shp=shp.to_crs('EPSG:4326')  # expresamos las geometrías en latitud y longitud WGS84.
    # documentamos aquí: centroide por proyeccion metrica, no grados geograficos.
    cent=gpd.GeoSeries(shp.to_crs(6372).centroid,crs=6372).to_crs(4326)  # calculamos centroides usando primero una proyección métrica.
    shp['latitud']=cent.y.to_numpy(); shp['longitud']=cent.x.to_numpy()  # añadimos latitud y longitud de cada parcela como variables explicativas.
    a=data.merge(shp.drop(columns=['geometry','Cultivo','CONJUNTO']),on='ID_POLIGONO',validate='one_to_one')  # relacionamos atributos geográficos y rendimientos por ID, no por posición.
    assert len(a)==197 and np.max(np.abs(a.AREA_HA-a.area_shp_ha))<0.011  # comprobamos una condición de integridad para detectar errores de datos.
    return a.sort_values('ID_POLIGONO').reset_index(drop=True)  # devolvemos el resultado calculado por esta función.




# ==================== INGENIERÍA DE VARIABLES SATELITALES ====================
def satellite_features(base,cloud_limit,base_ids):  # resumimos los índices de vegetación por sensor, parcela y mes.
    bas=pd.read_csv(find_one(base,'Conjunto_datos_BASICO*.csv'), low_memory=False)  # leemos las observaciones de Sentinel-2 y Landsat del dataset Básico.
    pro=pd.read_csv(find_one(base,'Conjunto_datos_PRO*.csv'), low_memory=False)  # leemos las observaciones de Planet del dataset PRO.
    frames=[]  # preparamos las observaciones satelitales de ambos CSV antes de unirlas.
    for raw in (bas,pro):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        raw['fecha_captura']=pd.to_datetime(raw.fecha_captura,format='%d/%m/%Y',errors='coerce')  # convertimos las fechas del satélite a fechas que Python pueda comparar.
        if raw.fecha_captura.isna().any(): raise ValueError('Fechas satelitales no parseables')  # comprobamos la condición antes de continuar con el procesamiento.
        raw=raw.loc[raw.fecha_captura.dt.year.eq(2025) & raw.fecha_captura.dt.month.isin(MESES)].copy()  # conservamos observaciones del periodo abril-octubre de 2025.
        frames.append(raw)  # acumulamos el subconjunto de observaciones de cada CSV.
    df=pd.concat(frames,ignore_index=True,sort=False)  # reunimos los registros de las dos fuentes y conservo el nombre del sensor.
    df['mes']=df.fecha_captura.dt.month  # extraemos el mes para construir variables temporales.
    df=df.loc[df['porcentaje_nubosidad'].le(cloud_limit)].copy()  # filtramos las observaciones que exceden el umbral de nubosidad elegido.
    df['ID_POLIGONO']=df.ID_POLIGONO.astype(str).str.strip()  # homogeneizamos los IDs de observaciones satelitales.
    allfeat=pd.DataFrame(index=pd.Index(base_ids,name='ID_POLIGONO'))  # inicializamos una tabla con una fila por parcela.
    for sensor,fields in SENSORS.items():  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        subset=df.loc[df.sensor.eq(sensor)].copy()  # separamos las observaciones por satélite para no confundir sus índices.
        actual=[f for f in fields if f in subset.columns]  # revisamos cuáles índices realmente están disponibles en este sensor.
        if subset.empty: continue  # comprobamos la condición antes de continuar con el procesamiento.
        # documentamos aquí: agregados robustos: mediana de indice de parcela por mes; fuentes nunca fusionadas directamente.
        aggregated=subset.groupby(['ID_POLIGONO','mes'])[actual].median()  # calculamos medianas mensuales de cada índice por parcela.
        wide=aggregated.unstack('mes')  # convertimos mediciones por mes en columnas del dataset final.
        wide.columns=[f'{sensor.replace("-","_")}_{col}_mes_{month:02d}' for col,month in wide.columns]  # pongomos a cada variable el nombre del sensor, índice y mes.
        allfeat=allfeat.join(wide,how='left')  # incorporamos las variables satelitales de este sensor por ID.
        for f in actual:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
            grp=subset.groupby('ID_POLIGONO')[f]  # reunimos las mediciones de un mismo índice por parcela.
            allfeat[f'{sensor.replace("-","_")}_{f}_ciclo_mediana']=grp.median()  # calculamos la mediana del índice durante abril-octubre por parcela.
            allfeat[f'{sensor.replace("-","_")}_{f}_ciclo_maximo']=grp.max()  # extraemos el máximo del índice observado en el ciclo por parcela.
        counts=subset.groupby('ID_POLIGONO').size()  # contamos cuántas observaciones satelitales claras tenemos por parcela.
        allfeat[f'{sensor.replace("-","_")}_n_observaciones_claras']=counts  # contamos observaciones conservadas tras filtrar nubes.
        monthly=subset.groupby(['ID_POLIGONO','mes']).size().unstack('mes')  # identificamos los meses que poseen observaciones disponibles.
        allfeat[f'{sensor.replace("-","_")}_n_meses_observados']=monthly.notna().sum(axis=1)  # contamos meses con alguna observación útil por parcela.
    return allfeat  # devolvemos el resultado calculado por esta función.




# ==================== EXTRACCIÓN DE VALORES DE RÁSTER ====================
def sample_tif_from_zip(z,nm,lon,lat):  # extraemos valores de los rásteres en la posición geográfica de las parcelas.
    dat=z.read(nm)  # leemos los bytes del archivo ráster comprimido.
    with MemoryFile(dat) as mem:  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
        with mem.open() as ds:  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
            trans=Transformer.from_crs('EPSG:4326',ds.crs,always_xy=True)  # construimos la transformación entre la ubicación de la parcela y el ráster.
            xx,yy=trans.transform(lon,lat)  # ubicamos los centros de las parcelas dentro del sistema del ráster.
            muestras = [p[0] for p in ds.sample(zip(xx, yy), masked=True)]  # obtenemos cada píxel y conservo la marca de datos ausentes del ráster.
            vals = np.array([np.nan if np.ma.is_masked(v) else float(v) for v in muestras], dtype=float)  # convertimos píxeles sin datos en NaN sin producir advertencias innecesarias.
            bad=(~np.isfinite(vals)) | (vals == ds.nodata) if ds.nodata is not None else ~np.isfinite(vals)  # identificamos valores inválidos o ausentes definidos por el ráster.
            if hasattr(ds,'nodatavals') and ds.nodatavals[0] is not None:  # comprobamos la condición antes de continuar con el procesamiento.
                bad=bad|(vals==ds.nodatavals[0])  # identificamos valores inválidos o ausentes definidos por el ráster.
            vals[bad]=np.nan  # convertimos las mediciones inválidas en NaN para tratarlas después.
            return vals  # devolvemos el resultado calculado por esta función.




# ==================== INGENIERÍA DE VARIABLES CLIMÁTICAS Y TOPOGRÁFICAS ====================
def raster_features(base,geodata):  # generamos las variables de lluvia, temperatura, altitud y pendiente.
    lons=geodata.longitud.to_numpy();lats=geodata.latitud.to_numpy()  # extraemos las coordenadas de cada parcela para muestrear rásteres.
    result=pd.DataFrame(index=geodata.ID_POLIGONO)  # preparamos una tabla de características climáticas con IDs de parcela.
    for source,pattern,prefix in [('prec','Reto_AgroCebada_CHIRPS*.zip','precipitacion'),  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                                   ('temp','Reto_AgroCebada_Temperatura*.zip','temperatura')]:  # cerramos la estructura de datos iniciada en las líneas anteriores.
        with zipfile.ZipFile(find_one(base,pattern)) as z:  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
            names=[x for x in z.namelist() if x.lower().endswith('.tif')]  # obtenemos la lista de archivos TIFF disponibles en el comprimido.
            collected=defaultdict(dict)  # organizamos las mediciones que corresponden a cada año y mes.
            for nm in names:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                mt=re.search(r'(202[2-5])_(\d{2})\.tif$',nm,re.IGNORECASE)  # reconozcomos el año y mes de cada ráster a partir de su nombre.
                if not mt:continue  # comprobamos la condición antes de continuar con el procesamiento.
                year,month=map(int,mt.groups())  # separamos el año y mes para construir las columnas climáticas.
                if source=='prec': variable='prec_mm'  # comprobamos la condición antes de continuar con el procesamiento.
                else: variable='tmax_c' if '/Tmax/' in nm else 'tmin_c'  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
                key=(variable,year)  # distinguimos variable climática y año de observación.
                collected[key][month]=sample_tif_from_zip(z,nm,lons,lats)  # extraemos y almaceno los valores del ráster para las parcelas.
            for variable in (['prec_mm'] if source=='prec' else ['tmin_c','tmax_c']):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                # documentamos aquí: 2025: cada mes, para estudiar etapas fenologicas, y suma o promedio de la temporada.
                series=[]  # preparamos una lista temporal para resumir los meses del ciclo.
                for month in MESES:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                    v=collected[(variable,2025)].get(month,np.full(len(geodata),np.nan))  # recuperamos el valor climático del mes para cada parcela.
                    result[f'{variable}_2025_mes_{month:02d}']=v  # agregamos una característica climática por parcela.
                    series.append(v)  # conservamos el mes para calcular el resumen de temporada.
                stack=np.array(series)  # convertimos los meses en una matriz numérica para resumirlos.
                with np.errstate(all='ignore'):  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
                    if variable=='prec_mm':  # comprobamos la condición antes de continuar con el procesamiento.
                        result[f'{variable}_2025_abril_octubre']=np.sum(stack,axis=0)  # agregamos una característica climática por parcela.
                    else: result[f'{variable}_2025_abril_octubre']=np.mean(stack,axis=0)  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
                # documentamos aquí: medias historicas 2022-24 disponibles (no datos futuros): ultimo ciclo incluido en 2025.
                historical=[]  # preparamos las mediciones históricas de 2022 a 2024.
                for year in (2022,2023,2024):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                    historical.extend([collected[(variable,year)][m] for m in MESES])  # agrupamos las mediciones climáticas de temporadas previas.
                with np.errstate(all='ignore'):  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
                    if variable=='prec_mm':  # comprobamos la condición antes de continuar con el procesamiento.
                        ar=np.array([sum(collected[(variable,y)][m] for m in MESES) for y in (2022,2023,2024)])  # calculamos la lluvia total de cada temporada histórica.
                        result[f'{variable}_promedio_estacional_2022_2024']=np.mean(ar,axis=0)  # agregamos una característica climática por parcela.
                    else: result[f'{variable}_promedio_estacional_2022_2024']=np.mean(np.array(historical),axis=0)  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
    with zipfile.ZipFile(find_one(base,'Reto_AgroCebada_Topografia*.zip')) as z:  # abrimos un recurso y me aseguro de cerrarlo al terminar esta operación.
        for desc,substr in [('elevacion_m','Elevacion_INEGI_CEM4_120m.tif'),('pendiente_grados','Pendiente_INEGI_CEM4_120m_grados.tif')]:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
            names=[x for x in z.namelist() if x.endswith(substr)]  # obtenemos la lista de archivos TIFF disponibles en el comprimido.
            if len(names)!=1: raise ValueError('No se encontro raster ' + desc)  # comprobamos la condición antes de continuar con el procesamiento.
            result[desc]=sample_tif_from_zip(z,names[0],lons,lats)  # incorporamos al dataset las variables topográficas de cada parcela.
    return result  # devolvemos el resultado calculado por esta función.




# ==================== INTEGRACIÓN DE CARACTERÍSTICAS POR PARCELA ====================
def build_features(base,cloud_limit,geo,raster):  # combinamos las diferentes fuentes en una fila por parcela.
    a=geo.set_index('ID_POLIGONO').copy()  # empleomos los identificadores de parcela como llave de relación.
    common=a[['AREA_HA','latitud','longitud','estado','municipio']].copy()  # seleccionamos superficie, coordenadas, estado y municipio.
    sat=satellite_features(base,cloud_limit,geo.ID_POLIGONO)  # construimos las características de los satélites con el umbral de nubes dado.
    all_x=common.join(raster).join(sat)  # integramos geografía, clima, topografía e índices por parcela.
    assert all_x.shape[0]==197 and all_x.index.is_unique  # comprobamos una condición de integridad para detectar errores de datos.
    return all_x  # devolvemos el resultado calculado por esta función.




# ==================== CONSTRUCCIÓN DE MODELOS INICIALES ====================
def make_model(X,model_kind='extratrees'):  # conectomos el preprocesamiento con un regresor de machine learning.
    cats=[x for x in ('estado','municipio') if x in X.columns]  # identificamos qué columnas representan categorías geográficas.
    nums=[x for x in X.columns if x not in cats]  # identificamos las variables numéricas que procesaré.
    # documentamos aquí: missing values (and categorical missing) estimated using training fold only.
    numerical=Pipeline([('imputar',SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True))])  # preparamos el relleno de faltantes usando solo cada grupo de entrenamiento.
    categorical=Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])  # preparamos la imputación y codificación de municipios y estados.
    transforms=[('numericas',numerical,nums)]  # configuromos el procesamiento de variables numéricas.
    if cats:transforms.append(('categoricas',categorical,cats))  # comprobamos la condición antes de continuar con el procesamiento.
    pre=ColumnTransformer(transforms,remainder='drop',sparse_threshold=0)  # unimos las transformaciones en una única operación reproducible.
    if model_kind=='ridge': reg=Ridge(alpha=30.0)  # comprobamos la condición antes de continuar con el procesamiento.
    else: reg=ExtraTreesRegressor(n_estimators=65,min_samples_leaf=2,max_features=0.85,random_state=SEED,n_jobs=1)  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
    return Pipeline([('preprocesamiento',pre),('regresion',reg)])  # devolvemos el resultado calculado por esta función.




# ==================== ESTRATEGIAS DE ENTRENAMIENTO REGIONAL ====================
def fit_strategy(model_kind,Xtr,ytr,states_tr,Xte,states_te,mode):  # entrenamos y predigo según la arquitectura regional seleccionada.
    base=make_model(Xtr,model_kind); base.fit(Xtr,ytr)  # construimos una instancia nueva del algoritmo con los datos del ensayo.
    pred=base.predict(Xte)  # estimamos los rendimientos del conjunto temporalmente reservado.
    if mode=='global': return pred  # comprobamos la condición antes de continuar con el procesamiento.
    if mode=='estado':  # comprobamos la condición antes de continuar con el procesamiento.
        for state in np.unique(states_te):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
            mask_tr=states_tr==state; mask_te=states_te==state  # seleccionamos las parcelas de entrenamiento de un solo estado.
            # documentamos aquí: tiny samples unstable; fallback global.
            if mask_tr.sum()<12:continue  # comprobamos la condición antes de continuar con el procesamiento.
            local=make_model(Xtr.loc[mask_tr],model_kind);local.fit(Xtr.loc[mask_tr],ytr[mask_tr])  # entrenamos un modelo local solo cuando hay ejemplos suficientes.
            pred[mask_te]=local.predict(Xte.loc[mask_te])  # usamos el modelo del estado correspondiente para las parcelas reservadas.
    elif mode=='ajuste':  # comprobamos una alternativa de configuración para decidir cómo procesarla.
        # documentamos aquí: shrunk correction based on in-sample residuals, conservative and fitted only on train fold.
        residual=ytr-base.predict(Xtr)  # calculamos errores de entrenamiento para la corrección regional experimental.
        for state in np.unique(states_te):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
            resid=residual[states_tr==state]  # seleccionamos los errores del entrenamiento en el estado actual.
            if not len(resid): continue  # comprobamos la condición antes de continuar con el procesamiento.
            correction=np.mean(resid)*len(resid)/(len(resid)+25)  # atenúomos una corrección regional para evitar un ajuste exagerado.
            pred[states_te==state]+=correction  # incorporamos una corrección regional atenuada a cada predicción.
    return pred  # devolvemos el resultado calculado por esta función.




# ==================== SELECCIÓN DE GRUPOS DE VARIABLES ====================
def get_feature_set(X,name):  # seleccionamos las familias de variables de cada experimento.
    geo=['AREA_HA','latitud','longitud','estado','municipio']  # definimos las cinco características geográficas básicas.
    if name=='geografia': cols=geo  # comprobamos la condición antes de continuar con el procesamiento.
    elif name=='satellite': cols=geo[:1]+[c for c in X.columns if c.startswith(('Sentinel_2','Planet','Landsat'))]  # comprobamos una alternativa de configuración para decidir cómo procesarla.
    elif name=='clima_geografia': cols=geo+[c for c in X.columns if c.startswith(('prec_mm','tmin_c','tmax_c','elevacion_m','pendiente_grados'))]  # comprobamos una alternativa de configuración para decidir cómo procesarla.
    elif name=='completo': cols=list(X.columns)  # comprobamos una alternativa de configuración para decidir cómo procesarla.
    else:raise ValueError(name)  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
    return X[cols].copy()  # devolvemos el resultado calculado por esta función.




# ==================== CONSTRUCCIÓN DE PARTICIONES DE VALIDACIÓN ====================
def folds(a,protocol):  # preparamos los grupos de entrenamiento y validación.
    if protocol=='parcelas':  # comprobamos la condición antes de continuar con el procesamiento.
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)  # distribuyomos parcelas en cinco particiones conservando representación por estado.
        return list(cv.split(a,a.estado))  # devolvemos el resultado calculado por esta función.
    # documentamos aquí: municipio no se repite entre train y validation. reparte 9 grupos en 5 bloques.
    keys=(a.estado.astype(str)+' | '+a.municipio.astype(str))  # asignomos una clave de grupo a cada combinación de estado y municipio.
    cv=GroupKFold(n_splits=5)  # reservamos municipios completos para evaluar generalización territorial.
    return list(cv.split(a,groups=keys))  # devolvemos el resultado calculado por esta función.




# ==================== EVALUACIÓN COMPARATIVA DE EXPERIMENTOS ====================
def benchmark(meta,features_by_cloud):  # ejecutamos comparaciones entre configuraciones y mido errores.
    y=meta.RENDIMIENTO_T_HA.to_numpy(float)  # extraemos los 138 rendimientos conocidos como objetivo de entrenamiento.
    state=meta.estado.to_numpy()  # almacenomos el estado de cada parcela para comparar estrategias regionales.
    rows=[];oofs=[]  # preparamos el registro de métricas de los experimentos.
    protocols=['parcelas','municipios']  # comparamos validación por parcelas y validación por municipios.
    # documentamos aquí: base: always training-fold mean; comparators per feature set.
    tasks=[(20,'geografia','global','extratrees'),  # declaramos las configuraciones iniciales de variables y modelos.
           (20,'satellite','global','extratrees'),  # cerramos la estructura de datos iniciada en las líneas anteriores.
           (20,'clima_geografia','global','extratrees')]  # cerramos la estructura de datos iniciada en las líneas anteriores.
    for cloud in (0,10,20,40):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        tasks.append((cloud,'completo','global','extratrees'))  # añadimos una configuración para comparar su desempeño.
    tasks.extend([(20,'completo','estado','extratrees'),(20,'completo','ajuste','extratrees')])  # incluyomos variantes de modelos independientes y ajustes por estado.
    # documentamos aquí: linear global regularized baseline (scaling ignored; prediction for comparison only).
    for protocol in protocols:  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        split=folds(meta,protocol)  # construimos los pliegues del protocolo de validación actual.
        for cloud,feat,arch,kind in [('NA','geografia','media','base')]+(tasks if protocol=='parcelas' else [t for t in tasks if t[0]==20 and t[1] in ('geografia','completo')]):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
            print('Evaluando',protocol,cloud,feat,arch,flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
            if arch=='media': X=None  # comprobamos la condición antes de continuar con el procesamiento.
            else: X=get_feature_set(features_by_cloud[cloud].loc[meta.ID_POLIGONO],feat)  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
            pred=np.full(len(meta),np.nan)  # reservamos un resultado para cada parcela etiquetada.
            fold_index=np.zeros(len(meta),dtype=int)  # registramos en qué partición se evaluó cada parcela.
            for k,(tr,te) in enumerate(split,1):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                fold_index[te]=k  # identificamos las parcelas apartadas en el pliegue actual.
                if arch=='media': pred[te]=np.mean(y[tr])  # comprobamos la condición antes de continuar con el procesamiento.
                else:  # aplicamos la alternativa correspondiente cuando no se cumplen las condiciones anteriores.
                    states_tr=state[tr]; states_te=state[te]  # separamos también los estados, respetando los mismos pliegues.
                    pred[te]=fit_strategy(kind,X.iloc[tr],y[tr],states_tr,X.iloc[te],states_te,arch)  # generamos una predicción fuera de entrenamiento para cada parcela de validación.
            assert np.isfinite(pred).all()  # comprobamos una condición de integridad para detectar errores de datos.
            summary=dict(protocolo=protocol,nubosidad_maxima_pct=cloud,variables=feat,arquitectura=arch,modelo=kind,  # guardamos el resumen estadístico del experimento completo.
                        mae=mean_absolute_error(y,pred),rmse=np.sqrt(mean_squared_error(y,pred)),r2=r2_score(y,pred),  # calculamos y conservo este valor para utilizarlo en las siguientes etapas.
                        n_parcelas=len(y))  # calculamos y conservo este valor para utilizarlo en las siguientes etapas.
            rows.append(summary)  # incorporamos métricas de esta configuración a la tabla comparativa.
            for parcel,st,mun,real,p,k in zip(meta.ID_POLIGONO,meta.estado,meta.municipio,y,pred,fold_index):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
                oofs.append(dict(ID_POLIGONO=parcel,estado=st,municipio=mun,real_t_ha=real,predicho_cv_t_ha=p,  # guardamos los errores por parcela para auditarlos individualmente.
                                 pliegue_cv=k,protocolo=protocol,nubosidad_maxima_pct=cloud,variables=feat,  # calculamos y conservo este valor para utilizarlo en las siguientes etapas.
                                 arquitectura=arch,modelo=kind,error_t_ha=p-real))  # calculamos y conservo este valor para utilizarlo en las siguientes etapas.
    return pd.DataFrame(rows),pd.DataFrame(oofs)  # devolvemos el resultado calculado por esta función.




# ==================== EJECUCIÓN PRINCIPAL DEL EXPERIMENTO ====================
def main():  # organizamos la ejecución completa desde la terminal.
    pa=argparse.ArgumentParser()  # preparamos el lector de parámetros de la terminal.
    pa.add_argument('--datos', type=Path, default=CARPETA_DATOS, help='Carpeta que contiene los archivos originales de FIRA')  # permitimos elegir la carpeta de originales desde Windows.
    pa.add_argument('--salida', type=Path, default=CARPETA_SALIDA, help='Carpeta para guardar nuevas métricas y predicciones')  # permitimos escoger la carpeta donde escribo resultados.
    pa.add_argument('--solo-verificar', action='store_true', help='Compruebo los archivos originales sin entrenar modelos')  # habilitamos una prueba corta que revisa archivos sin entrenar.
    args=pa.parse_args()  # leemos las opciones introducidas al ejecutar el programa.
    verificar_insumos(args.datos)  # revisamos los nombres de los archivos antes del análisis costoso.
    if args.solo_verificar:  # comprobamos la condición antes de continuar con el procesamiento.
        return  # devolvemos el resultado calculado por esta función.
    out=args.salida.resolve()  # calculamos la ruta absoluta que utilizaré para guardar resultados.
    datos=args.datos.resolve()  # calculamos la ruta absoluta que contiene los originales para protegerlos.
    if out == datos or datos in out.parents:  # impedimos que el programa escriba resultados dentro de los datos originales.
        raise ValueError('No se permite guardar resultados en la carpeta de originales FIRA.')  # detenemos una configuración que pondría en riesgo los originales.
    out.mkdir(parents=True,exist_ok=True)  # creamos la carpeta local donde guardaré exclusivamente resultados nuevos.
    geo=read_geodata(args.datos)  # validamos las geometrías y las etiquetas por identificador.
    print('Geometria verificada:',len(geo),'parcelas; etiquetas conocidas:',geo.RENDIMIENTO_T_HA.notna().sum(),flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    rast=raster_features(args.datos,geo)  # extraemos las series climáticas y características topográficas.
    features_by_cloud={}  # preparamos distintas matrices según el porcentaje de nubosidad permitido.
    for cloud in (0,10,20,40):  # recorremos los elementos de este conjunto para procesarlos uno por uno.
        features_by_cloud[cloud]=build_features(args.datos,cloud,geo,rast)  # creamos una matriz por parcela para este umbral de nubosidad.
        features_by_cloud[cloud].to_csv(out/f'caracteristicas_nubes_{cloud}.csv',index=True)  # exportamos la matriz derivada dentro del proyecto para repetir estudios.
        print('Nubosidad',cloud,'features',features_by_cloud[cloud].shape,flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    train=geo[geo.CONJUNTO.eq('ENTRENAMIENTO')].reset_index(drop=True)  # separamos las 138 parcelas con rendimiento conocido.
    test=geo[geo.CONJUNTO.eq('PREDICCION')].reset_index(drop=True)  # separamos las 59 parcelas de rendimiento oculto.
    assert train.RENDIMIENTO_T_HA.notna().all() and test.RENDIMIENTO_T_HA.isna().all()  # comprobamos una condición de integridad para detectar errores de datos.
    scores,oof=benchmark(train,features_by_cloud)  # evaluamos candidatos y guardo predicciones fuera de muestra.
    scores=scores.sort_values(['protocolo','rmse']).reset_index(drop=True)  # ordenamos modelos según protocolo y RMSE.
    scores.to_csv(out/'comparacion_modelos_cv.csv',index=False)  # exportamos las métricas en un CSV para auditarlas.
    oof.to_csv(out/'predicciones_validacion_cruzada.csv',index=False)  # exportamos las predicciones usadas para medir errores.
    print('\nResultados mejores por protocolo:',flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    print(scores.groupby('protocolo').head(8).to_string(index=False),flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    # documentamos aquí: model selection based only on cross-validated errors of known labeled parcels.
    eligible=scores.loc[(scores.protocolo=='parcelas') & (scores.modelo=='extratrees')]  # limitamos la selección provisional a los experimentos comparables.
    best=eligible.sort_values('rmse').iloc[0]  # seleccionamos provisionalmente el menor RMSE de validación.
    cl=int(best.nubosidad_maxima_pct);fs=best.variables;arch=best.arquitectura  # recuperamos nubosidad, variables y estrategia elegidas solo como propuesta provisional.
    chosen=get_feature_set(features_by_cloud[cl],fs)  # reconstruyomos las variables usadas por la configuración seleccionada.
    predict=fit_strategy('extratrees',chosen.loc[train.ID_POLIGONO],train.RENDIMIENTO_T_HA.to_numpy(),  # entrenamos con las 138 etiquetas y predigo las 59 parcelas ocultas.
                         train.estado.to_numpy(),chosen.loc[test.ID_POLIGONO],test.estado.to_numpy(),arch)  # cerramos correctamente la expresión iniciada en líneas anteriores.
    if not np.isfinite(predict).all():raise ValueError('Predicciones no finitas')  # comprobamos la condición antes de continuar con el procesamiento.
    output=test[['ID_POLIGONO','estado','municipio','AREA_HA']].copy()  # preparamos un registro por parcela para entregar sus predicciones.
    output['RENDIMIENTO_PREDICHO_T_HA']=np.round(predict,4)  # guardamos las toneladas por hectárea estimadas.
    output['PRODUCCION_ESTIMADA_T']=np.round(predict*test.AREA_HA.to_numpy(),4)  # multiplicomos toneladas por hectárea por hectáreas de la parcela.
    assert len(output)==59 and output.ID_POLIGONO.nunique()==59  # comprobamos una condición de integridad para detectar errores de datos.
    output.to_csv(out/'predicciones_experimentales_59_NO_FINALES.csv',index=False)  # exportamos el CSV experimental y aclaro en su nombre que no es final.
    chosen.to_csv(out/'matriz_variables_197_parcelas.csv',index=True)  # guardamos las variables de las 197 parcelas en CSV.
    conf={'estado':'PRELIMINAR_NO_ENTREGA', 'fecha_generacion_utc':datetime.now(timezone.utc).isoformat(),  # documentamos la configuración elegida y sus advertencias metodológicas.
          'parcelas_totales':len(geo),'etiquetadas':len(train),'ocultas':len(test),  # cerramos la estructura de datos iniciada en las líneas anteriores.
          'metodo_seleccion':'min_rmse_cv_por_parcelas_5_folds','modelo':'extratrees','arquitectura':arch,  # cerramos la estructura de datos iniciada en las líneas anteriores.
          'variables':fs,'nubosidad_max_pct':cl,'rmse_cv_parcelas':float(best.rmse),'mae_cv_parcelas':float(best.mae),  # cerramos la estructura de datos iniciada en las líneas anteriores.
          'r2_cv_parcelas':float(best.r2),'aviso':'CV usado tambien para seleccionar hiperparametros: estimacion optimista; necesita validacion adicional',  # cerramos la estructura de datos iniciada en las líneas anteriores.
          'nota':'Estos 59 resultados NO son predicciones oficiales definitivas; aun falta analisis de estabilidad, pruebas, documentacion.'}  # cerramos la estructura de datos iniciada en las líneas anteriores.
    (out/'configuracion_experimento.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf-8')  # documentamos los parámetros y advertencias metodológicas del experimento.
    print('\nMEJOR EXPERIMENTO SOLO PROVISIONAL:',conf,flush=True)  # mostramos en la terminal el avance o el resultado de la operación.
    print('Archivos exportados a:',out,flush=True)  # mostramos en la terminal el avance o el resultado de la operación.

if __name__=='__main__':main()  # comprobamos la condición antes de continuar con el procesamiento.
