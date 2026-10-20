"""Presentamos AgroBYTE 3.0: predicción real, cartografía y simulación responsable."""


# ==================== 1. IMPORTACIÓN DE BIBLIOTECAS ====================
from __future__ import annotations  # Activamos anotaciones modernas sin cambiar cálculos.
import json  # Leemos los polígonos y el manifiesto en formatos transparentes.
from pathlib import Path  # Construimos rutas internas de forma portable.
import numpy as np  # Preparamos cálculos y conversiones numéricas.
import pandas as pd  # Consultamos métricas y preparamos descargas ordenadas.
import plotly.express as px  # Creamos gráficas interactivas para datos científicos.
import streamlit as st  # Construimos una aplicación web pública desde Python.
from codigo.entrega_v3 import RAIZ, cargar_entrega, predecir_parcela, simular  # Reutilizamos funciones verificables y testeables.


# ==================== 2. CONFIGURACIÓN GENERAL Y TEMA OSCURO ====================
st.set_page_config(page_title='AgroBYTE 3.0 | FIRA 2026',page_icon='🌾',layout='wide',initial_sidebar_state='expanded')  # Preparamos ventana y navegación.
if 'oscuro' not in st.session_state:  # Conservamos la elección del tema durante la sesión.
    st.session_state.oscuro = False  # Iniciamos en modo claro por defecto.
oscuro = st.sidebar.toggle('🌙 Modo oscuro',key='oscuro')  # Permitir alternar colores sin recargar el proyecto.
bg,card,fore,borde,verde = ('#0d1720','#172633','#eff5ed','#304653','#7cdba2') if oscuro else ('#f6f9f3','#ffffff','#19362d','#dae5dd','#236746')  # Definimos dos paletas legibles.
st.markdown(f'''<style>
.stApp, [data-testid="stAppViewContainer"]{{background:{bg};color:{fore};}}
[data-testid="stSidebar"]{{background:{card};}}
[data-testid="stMetric"]{{background:{card};border:1px solid {borde};border-radius:14px;padding:15px;}}
[data-testid="stMetricLabel"]{{color:{fore};}}
[data-testid="stMarkdownContainer"] p{{line-height:1.52;}}
.block-container{{padding-top:1.6rem;max-width:1480px;}}
h1,h2,h3{{color:{fore}!important;}}
[data-testid="stMarkdownContainer"]{{color:{fore};}}
[data-testid="stWidgetLabel"]{{color:{fore};}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] small, [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{{color:{fore}!important;opacity:0.88;}}
[data-testid="stHeader"]{{background:{bg};color:{fore};}}
[data-testid="stToolbar"]{{background:{bg};}}
a{{color:{verde};}}
</style>''',unsafe_allow_html=True)  # Estilizamos las tarjetas sin modificar librerías ni resultados.


# ==================== 3. LECTURA ÚNICA Y SEGURA DE LA ENTREGA ====================
@st.cache_resource(show_spinner='Comprobando modelo y 59 predicciones…')
def recuperar_entrega():  # Evitamos cargar 150 árboles con cada interacción del usuario.
    return cargar_entrega()  # Ejecutamos las comprobaciones de firmas y reproducción exacta.

try:  # Mostramos mensajes comprensibles si la entrega está incompleta.
    tabla,manifiesto,modelo,avisos = recuperar_entrega()  # Recuperamos las 59 parcelas verificadas y el modelo real.
except Exception as exc:  # Detenemos cualquier presentación con inferencia no verificada.
    st.error(f'No se puede iniciar AgroBYTE con resultados verificados: {exc}')  # Indicamos el problema y no inventamos valores.
    st.info('Consultamos README.md → Diagnóstico de instalación. Verificamos versiones, dependencias y archivos SHA-256.')  # Orientamos al equipo.
    st.stop()  # Evitamos continuar con datos incompletos.


# ==================== 4. FUNCIONES DE FORMATO Y EVIDENCIA ====================
def pesos(x: float) -> str:  # Mostramos cantidades en pesos mexicanos de manera uniforme.
    return f'${x:,.2f} MXN'  # Evitamos confundir monedas o unidades.


def grafica_plotly(fig):  # Aplicamos un tema visual coherente a nuestros gráficos.
    fig.update_layout(template='plotly_dark' if oscuro else 'plotly_white',height=410,margin=dict(l=15,r=15,t=55,b=15))  # Adaptamos al modo claro u oscuro.
    st.plotly_chart(fig,use_container_width=True)  # Mostramos el gráfico interactivo.


def leer_evidencia(rel: str) -> pd.DataFrame:  # Leemos únicamente archivos de evaluación empírica conservados.
    ruta=RAIZ / 'evidencia' / rel  # Identificamos el archivo interno del proyecto.
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()  # Permitimos seguir si falta evidencia opcional.


# ==================== 5. NAVEGACIÓN Y CONTEXTO FIRA ====================
st.sidebar.markdown('## 🌾 AgroBYTE 3.0')  # Identificamos la aplicación ante el jurado.
st.sidebar.caption('Reto AgroCebada 2026 · Prototipo académico')  # Distinguimos la herramienta de un servicio oficial de FIRA.
pagina=st.sidebar.radio('Secciones',['Resumen','Mapa y parcelas','Ejecutar modelo','Simulador económico','Ciencia de datos','Documentación y descargas'],index=0)  # Organizamos el flujo por objetivos.
st.sidebar.divider()  # Separamos navegación de procedencia.
st.sidebar.caption('Modelo: Extra Trees geográfico · 138 etiquetas · 59 parcelas de predicción')  # Resumimos sin afirmar exactitud oculta.
st.sidebar.caption('No es una herramienta oficial de aprobación crediticia de FIRA.')  # Delimitamos su uso financiero.

st.title('🌾 AgroBYTE · Inteligencia agrícola')  # Titulamos el producto.
st.caption('Predicción del rendimiento de cebada en Puebla, Hidalgo y Tlaxcala | Reto FIRA AgroCebada 2026')  # Mostramos alcance geográfico.
if avisos:  # Avisamos si hubo incompatibilidades de bibliotecas al cargar.
    st.info('El modelo fue guardado con otra versión de scikit-learn; **reprodujo correctamente las 59 predicciones**. Para auditoría estricta utilizamos las versiones de entrenamiento documentadas.')  # No confundimos reproducción local con equivalencia universal.
st.warning('Resultados **estimados** del ciclo 2025. El rendimiento real de las 59 parcelas no se conoce. No representa un pronóstico validado antes de sembrar.')  # Evitamos interpretaciones de riesgo.


# ==================== 6. PÁGINA: RESUMEN PRODUCTIVO ====================
if pagina == 'Resumen':  # Presentamos el estado del prototipo en una sola pantalla.
    area = float(tabla.AREA_HA.sum())  # Sumamos las hectáreas de las 59 parcelas.
    toneladas=float(tabla.PRODUCCION_ESTIMADA_T.sum())  # Calculamos la producción agregada estimada.
    a,b,c,d=st.columns(4)  # Creamos indicadores principales de fácil lectura.
    a.metric('Parcelas de evaluación',len(tabla))  # Mostramos la cobertura completa.
    b.metric('Superficie',f'{area:,.2f} ha')  # Utilizamos hectáreas, nunca m².
    c.metric('Rendimiento ponderado',f'{toneladas/area:.3f} t/ha')  # Calculamos toneladas divididas entre hectáreas.
    d.metric('Producción estimada',f'{toneladas:,.2f} t')  # Mostramos el volumen previsto, no cosechado.
    st.subheader('Producción estimada por estado')  # Organizamos información de negocio.
    x=tabla.groupby('estado',as_index=False).agg(produccion_t=('PRODUCCION_ESTIMADA_T','sum'),area_ha=('AREA_HA','sum'),parcelas=('ID_POLIGONO','count'))  # Agregamos sin mezclar unidades.
    grafica_plotly(px.bar(x,x='estado',y='produccion_t',color='estado',labels={'produccion_t':'Producción estimada (toneladas)','estado':'Estado'},title='Escenario productivo por estado'))  # Mostramos diferencias territoriales.
    st.info('El principal valor de AgroBYTE es convertir características conocidas de una parcela en una estimación reproducible y en escenarios económicos transparentes.')  # Comunicamos utilidad.
    st.markdown('**Cómo explorar la solución:** Mapa y parcelas → Ejecutar modelo → Simulador económico → Ciencia de datos.')  # Guiamos una demostración de menos de cinco minutos.


# ==================== 7. PÁGINA: MAPA GEOGRÁFICO Y PARCELAS ====================
elegida_global=st.session_state.get('parcela_actual',str(tabla.ID_POLIGONO.iloc[0]))  # Recuperamos la selección sin cambiar el nombre entre páginas.
if pagina == 'Mapa y parcelas':  # Permitimos localizar parcelas y estudiar patrones espaciales.
    x1,x2=st.columns([1,1.5])  # Separamos filtros y ficha de mapa interactivo.
    with x1:  # Situamos los filtros en una columna de consulta.
        estados=st.multiselect('Filtrar por estado',sorted(tabla.estado.unique()),default=sorted(tabla.estado.unique()))  # Permitimos comparar uno o varios estados.
        municipios=sorted(tabla.loc[tabla.estado.isin(estados),'municipio'].unique())  # Limitamos municipios a los estados elegidos.
        municipio=st.selectbox('Municipio',['Todos',*municipios])  # Facilitamos seleccionar el lugar.
        filtradas=tabla.loc[tabla.estado.isin(estados) & (tabla.municipio.eq(municipio) if municipio != 'Todos' else True)].copy()  # Aplicamos filtro sin modificar la base.
        if filtradas.empty:  # Atendemos filtros sin resultados.
            st.info('No hay parcelas con estos filtros.')  # Evitamos gráfica vacía confusa.
        else:  # Mostramos una parcela válida y su ficha.
            opciones_mapa=filtradas.ID_POLIGONO.tolist()  # Identificamos las parcelas realmente visibles tras los filtros.
            indice_mapa=opciones_mapa.index(elegida_global) if elegida_global in opciones_mapa else 0  # Mantenemos la selección compartida cuando sigue visible.
            seleccionado=st.selectbox('Seleccionar parcela',opciones_mapa,index=indice_mapa,key='selector_mapa')  # Facilitamos la selección por ID.
            st.session_state.parcela_actual=seleccionado  # Compartimos la selección con los demás módulos.
            fila=filtradas.set_index('ID_POLIGONO').loc[seleccionado]  # Consultamos su estado y superficie.
            st.metric('Rendimiento predicho',f"{fila.RENDIMIENTO_PREDICHO_T_HA:.3f} t/ha")  # Mostramos la magnitud principal.
            st.metric('Producción estimada',f"{fila.PRODUCCION_ESTIMADA_T:.2f} t")  # Mostramos superficie por rendimiento.
            st.caption(f"{fila.municipio}, {fila.estado} · {fila.AREA_HA:.2f} ha")  # Identificamos la región.
    with x2:  # Mostramos polígonos de parcelas con sus límites reales.
        try:  # Evitamos que un fallo de teselas interrumpa las demás páginas.
            import folium  # Creamos polígonos geográficos interactivos sin token privado.
            from streamlit_folium import st_folium  # Incorporamos el mapa en la interfaz Streamlit.
            geojson=json.loads((RAIZ/'datos'/'poligonos_59.geojson').read_text(encoding='utf-8'))  # Leemos la geometría derivada de las 59 parcelas.
            visible=set(filtradas.ID_POLIGONO)  # Seleccionamos únicamente parcelas filtradas.
            propiedades=filtradas.set_index('ID_POLIGONO')  # Indexamos rendimientos por ID para colorear.
            features=[]  # Preparamos un GeoJSON compacto con propiedades de visualización.
            for feature in geojson['features']:  # Recorremos cada polígono conservado.
                f=feature.copy()  # Creamos una copia para no modificar datos fuente.
                id_=f['properties']['ID_POLIGONO']  # Obtenemos la llave de la parcela.
                if id_ not in visible:  # Omitimos parcelas fuera del filtro.
                    continue  # Continuamos con la siguiente geometría.
                val=float(propiedades.loc[id_,'RENDIMIENTO_PREDICHO_T_HA'])  # Recuperamos rendimiento asociado.
                f['properties']={'Parcela':id_,'Rendimiento_t_ha':round(val,3),'Estado':str(propiedades.loc[id_,'estado'])}  # Mostramos campos mínimos.
                features.append(f)  # Incorporamos el polígono a la vista geográfica.
            # ==================== CARTOGRAFÍA SIN CLAVES COMERCIALES ====================
            # Ofrecemos OpenStreetMap para consulta conectada y un modo offline con los polígonos, sin claves API.
            fondo=st.radio('Capa del mapa',['OpenStreetMap (sin clave)','Solo parcelas (sin conexión)'],horizontal=True,key='capa_mapa')  # Elegimos una alternativa pública o completamente local.
            capa_osm=fondo=='OpenStreetMap (sin clave)'  # Identificamos si necesitamos solicitar teselas por internet.
            m=folium.Map(location=[19.78,-98.40],zoom_start=9,tiles='OpenStreetMap' if capa_osm else None,control_scale=True)  # Evitamos los proveedores que actualmente piden API KEY.
            if not capa_osm:  # Informamos que el fondo se ha desactivado deliberadamente.
                st.caption('Vista sin conexión: mostramos los polígonos sin cartografía de fondo.')  # No confundimos la falta de mapa base con una falla del modelo.
            def color_terreno(valor):  # Clasificamos rendimientos sin cambiar las estimaciones.
                return '#d0e2a9' if valor < 3.3 else ('#72bc89' if valor < 4.2 else '#237a51')  # Utilizamos cortes ilustrativos y visibles.
            if features:  # Evitamos construir un componente GeoJSON vacío.
                folium.GeoJson({'type':'FeatureCollection','features':features},style_function=lambda f:{'fillColor':color_terreno(f['properties']['Rendimiento_t_ha']),'color':'#164432','weight':2,'fillOpacity':0.65},highlight_function=lambda f:{'color':'#ff9c3b','weight':4,'fillOpacity':0.85},tooltip=folium.GeoJsonTooltip(fields=['Parcela','Estado','Rendimiento_t_ha'],aliases=['ID:','Estado:','t/ha:'])).add_to(m)  # Mostramos tooltips al pasar el puntero.
                limites=[filtradas.latitud.min(),filtradas.longitud.min(),filtradas.latitud.max(),filtradas.longitud.max()]  # Delimitamos los centros de los terrenos visibles.
                m.fit_bounds([[float(limites[0])-.006,float(limites[1])-.006],[float(limites[2])+.006,float(limites[3])+.006]])  # Centramos el mapa en el conjunto filtrado.
            st_folium(m,height=540,returned_objects=[],key='visor_parcelas_fira')  # Renderizamos el mapa con desplazamiento y acercamiento.
            st.caption('Escala ilustrativa: claro < 3.3 t/ha, verde intermedio de 3.3 a 4.2 y verde intenso ≥ 4.2. Los polígonos provienen de FIRA y el rendimiento es **estimado**, no observado. Si no carga el fondo, elegimos «Solo parcelas».')  # Describimos la fuente.
        except Exception as exc:  # Comunicamos errores sin falsificar ubicaciones.
            st.warning(f'El mapa no pudo cargarse: {exc}')  # Damos visibilidad al problema.
    st.subheader('Tabla de predicciones')  # Presentamos los resultados en formato descargable.
    st.dataframe(filtradas[['ID_POLIGONO','estado','municipio','AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']],hide_index=True,use_container_width=True)  # Permitimos ordenar y consultar.


# ==================== 8. PÁGINA: INFERENCIA REAL CON EXTRA TREES ====================
if pagina == 'Ejecutar modelo':  # Permitimos al jurado activar un cálculo real de predicción.
    st.subheader('Ejecutar el pipeline Extra Trees')  # Diferenciamos cálculo real de CSV precalculado.
    st.write('Elegimos una de las 59 parcelas oficiales. El modelo cargado ejecuta `predict()` utilizando superficie, latitud, longitud, estado y municipio.')  # Explicamos el contrato.
    opciones=tabla.ID_POLIGONO.tolist()  # Restringimos las entradas al conjunto científicamente auditado.
    indice=opciones.index(elegida_global) if elegida_global in opciones else 0  # Recuperamos selección anterior cuando existe.
    elegida=st.selectbox('Parcela oficial',opciones,index=indice,key='selector_prediccion')  # Permitimos seleccionar cualquier ID oculto.
    st.session_state.parcela_actual=elegida  # Conservamos la parcela para el simulador.
    fila=tabla.set_index('ID_POLIGONO').loc[elegida]  # Recuperamos las cinco variables del modelo.
    st.table(pd.DataFrame({'Variable':['Superficie','Latitud','Longitud','Estado','Municipio'],'Valor':[f'{fila.AREA_HA:.4f} ha',f'{fila.latitud:.6f}',f'{fila.longitud:.6f}',fila.estado,fila.municipio]}))  # Hacemos visibles los datos usados.
    if st.button('🌱 Calcular rendimiento con Extra Trees',type='primary'):  # Ejecutamos el modelo solo tras la acción del usuario.
        pred,ton=predecir_parcela(modelo,tabla,elegida)  # Invocamos el pipeline guardado y calculamos toneladas.
        x,y=st.columns(2)  # Distribuimos los resultados en dos indicadores.
        x.metric('Predicción ejecutada',f'{pred:.4f} t/ha')  # Mostramos rendimiento por hectárea.
        y.metric('Producción estimada',f'{ton:.3f} t')  # Mostramos toneladas totales.
        st.success('Inferencia completada con el modelo entrenado. La misma predicción está registrada en el CSV firmado.')  # Validamos coherencia.
    st.info('La ejecución está restringida a las 59 parcelas del reto. Dibujar una parcela nueva o cambiar el mes de siembra requiere un modelo diferente y una validación adicional.')  # Reconocemos límites.


# ==================== 9. PÁGINA: RENTABILIDAD Y SENSIBILIDAD ====================
if pagina == 'Simulador económico':  # Complementamos la predicción con una decisión financiera hipotética.
    st.subheader('Simulador de rentabilidad por parcela')  # Presentamos una herramienta educativa, no una aprobación crediticia.
    opciones=tabla.ID_POLIGONO.tolist()  # Ofrecemos únicamente parcelas del conjunto auditado.
    elegida=st.selectbox('Parcela',opciones,index=opciones.index(elegida_global) if elegida_global in opciones else 0,key='selector_finanzas')  # Elegimos parcela oficial.
    st.session_state.parcela_actual=elegida  # Sincronizamos la selección.
    fila=tabla.set_index('ID_POLIGONO').loc[elegida]  # Recuperamos rendimiento y superficie del modelo.
    area=float(fila.AREA_HA)  # Conservamos hectáreas oficiales.
    rendimiento=float(fila.RENDIMIENTO_EN_VIVO_T_HA)  # Utilizamos una inferencia realmente ejecutada.
    st.caption(f"{fila.estado} · {fila.municipio} · {area:.2f} ha · predicción de referencia {rendimiento:.3f} t/ha")  # Contextualizamos el escenario.
    uno,dos=st.columns(2)  # Separamos precio y costos por hectárea.
    with uno:  # Mostramos variables comerciales editables.
        precio=st.number_input('Precio de venta supuesto (MXN/t)',min_value=0.0,value=0.0,step=100.0,help='No se consulta automáticamente el precio del mercado; introducimos el valor que conoce el usuario.')  # Evitamos inventar precios.
        variacion=st.slider('Variación del rendimiento para sensibilidad (%)',-35,35,0,5)  # Planteamos un escenario hipotético, no un intervalo de confianza.
        credito=st.number_input('Crédito simulado (MXN)',min_value=0.0,value=0.0,step=1000.0)  # Permitimos ilustrar costo de financiamiento.
        tasa=st.number_input('Tasa anual simple (%)',min_value=0.0,value=0.0,step=0.5)  # Registramos una tasa elegida por el usuario.
        meses=st.number_input('Plazo simulado (meses)',min_value=0,max_value=120,value=0,step=1)  # Configuramos horizonte.
    with dos:  # Pedimos costos transparentes por hectárea.
        semilla=st.number_input('Semilla (MXN/ha)',min_value=0.0,value=0.0,step=100.0)  # Solicitamos costo de semilla.
        fertilizante=st.number_input('Fertilizante (MXN/ha)',min_value=0.0,value=0.0,step=100.0)  # Solicitamos costo de fertilización.
        mano=st.number_input('Mano de obra (MXN/ha)',min_value=0.0,value=0.0,step=100.0)  # Solicitamos costo laboral.
        otro=st.number_input('Otros costos (MXN/ha)',min_value=0.0,value=0.0,step=100.0)  # Permitimos capturar costos complementarios.
    if precio<=0:  # Advertimos que no se introdujeron valores económicos suficientes.
        st.info('Para obtener un resultado económico, ingresamos un precio realista y los costos que correspondan a la parcela. Los valores empiezan en cero para no inventar precios.')  # Orientamos al usuario.
    escenario=simular(area,rendimiento*(1+variacion/100),precio,semilla,fertilizante,mano,otro,credito,tasa,meses)  # Calculamos el escenario con los supuestos elegidos.
    a,b,c,d=st.columns(4)  # Mostramos indicadores de interés para la planeación.
    a.metric('Producción escenario',f"{escenario['produccion_t']:,.2f} t")  # Calculamos toneladas bajo variación supuesta.
    b.metric('Ingresos brutos',pesos(escenario['ingreso_mxn']))  # Calculamos ingreso antes de costos.
    c.metric('Costos + intereses',pesos(escenario['costo_mxn']+escenario['interes_simple_mxn']))  # Mostramos costos declarados.
    d.metric('Margen estimado',pesos(escenario['margen_estimado_mxn']))  # Mostramos resultado antes de otros costos e impuestos.
    punto=escenario['rendimiento_equilibrio_t_ha']  # Recuperamos el rendimiento mínimo para cubrir supuestos.
    st.write(f"**Punto de equilibrio:** {punto:.2f} t/ha" if punto is not None else '**Punto de equilibrio:** requiere superficie y precio mayores a cero.')  # Explicamos la condición de rentabilidad.
    estados=[]  # Preparamos escenarios para comparar sensibilidades de rendimiento.
    for nombre,porcentaje in [('Conservador',-0.15),('Base',0),('Optimista',0.15)]:  # Definimos variaciones ilustrativas de ±15%.
        sim=simular(area,rendimiento*(1+porcentaje),precio,semilla,fertilizante,mano,otro,credito,tasa,meses)  # Recalculamos montos bajo cada supuesto.
        estados.append({'escenario':nombre,'rendimiento_t_ha':rendimiento*(1+porcentaje),'produccion_t':sim['produccion_t'],'margen_mxn':sim['margen_estimado_mxn']})  # Documentamos resultado comparable.
    tabla_sim=pd.DataFrame(estados)  # Organizamos la simulación en una tabla reproducible.
    st.dataframe(tabla_sim,hide_index=True,use_container_width=True)  # Mostramos escenarios financieros.
    st.download_button('Descargar escenarios CSV',tabla_sim.to_csv(index=False).encode('utf-8-sig'),file_name=f'agrobyte_escenarios_{elegida}.csv',mime='text/csv')  # Permitimos exportar sin guardar información del usuario en servidor.
    st.warning('Simulación hipotética. No incorpora necesariamente transporte, seguros, impuestos, riesgo de precio, calendario de flujo ni requisitos de crédito. Los escenarios ±15% no son intervalos de predicción.')  # Evitamos decisiones financieras automáticas.


# ==================== 10. PÁGINA: EVIDENCIA Y PREGUNTAS GUÍA FIRA ====================
if pagina == 'Ciencia de datos':  # Sustentamos la selección del modelo ante jurado técnico.
    st.subheader('¿Por qué Extra Trees?')  # Introducimos la justificación experimental.
    a,b,c=st.columns(3)  # Mostramos métricas anidadas derivadas de nuestro manifiesto.
    met=manifiesto['metricas_cv_anidada']  # Tomamos las cifras verificadas, nunca el rendimiento oculto.
    a.metric('RMSE de validación',f"{met['rmse_t_ha']:.3f} t/ha")  # Diferenciamos estimación de error.
    b.metric('MAE de validación',f"{met['mae_t_ha']:.3f} t/ha")  # Mostramos promedio de error absoluto.
    c.metric('R² de validación',f"{met['r2']:.3f}")  # Mostramos capacidad explicativa comparada con referencia.
    st.caption('Las métricas se calcularon con las 138 parcelas de entrenamiento en validación; FIRA conserva las etiquetas de las 59 parcelas ocultas.')  # Delimitamos inferencias.
    a=leer_evidencia('fase3/ranking.csv')  # Recuperamos ranking exploratorio por parcelas.
    if not a.empty:  # Dibujamos solo si existe la evidencia local.
        st.subheader('Comparación de modelos')  # Presentamos una investigación honesta.
        st.dataframe(a,hide_index=True,use_container_width=True)  # Conservamos los nombres y cifras originales.
    st.subheader('Preguntas guía de FIRA: índices, clima y combinaciones')  # Incluimos análisis con bonificación potencial.
    columnas=st.tabs(['Combinaciones','Índices satelitales','Variables climáticas'])  # Dividimos preguntas de la convocatoria.
    with columnas[0]:  # Investigamos combinaciones, sin alterar selección final.
        r=leer_evidencia('fase5/ablaciones.csv')  # Recuperamos validación de grupos de variables.
        if not r.empty:  # Dibujamos resultados medidos.
            st.dataframe(r,hide_index=True,use_container_width=True)  # Mostramos RMSE de geografía, clima, satélite y combinaciones.
        st.write('El mejor RMSE exploratorio correspondió a geografía. Añadir todas las características no demostró una mejora consistente.')  # Explicamos decisión.
    with columnas[1]:  # Investigamos importancia por permutación de índices.
        r=leer_evidencia('fase5/importancia_indices.csv')  # Recuperamos la tabla original.
        if not r.empty:  # Evitamos mostrar gráficos ficticios.
            st.dataframe(r,hide_index=True,use_container_width=True)  # Permite examinar LAI, EVI y demás índices.
        st.caption('LAI y EVI destacaron casi empatados en el modelo auxiliar solo satelital. La importancia depende del agrupamiento y de la correlación.')  # Evitamos afirmar causalidad.
    with columnas[2]:  # Investigamos variables meteorológicas.
        r=leer_evidencia('fase5/importancia_clima.csv')  # Recuperamos importancia de grupos climáticos.
        if not r.empty:  # Presentamos resultados derivados de validación.
            st.dataframe(r,hide_index=True,use_container_width=True)  # Mostramos precipitación y temperaturas.
        st.caption('La precipitación CHIRPS mostró la mayor importancia en el modelo solo climático. No implica causalidad.')  # Delimitamos conclusión.
    st.info('Las características satelitales de 2025 son del ciclo histórico observado; no demostramos predicción antes de sembrar ni para nuevas temporadas.')  # Discriminamos retrospectivo y prospectivo.


# ==================== 11. PÁGINA: DOCUMENTACIÓN Y DESCARGAS ====================
if pagina == 'Documentación y descargas':  # Permitimos reproducir, verificar y descargar el resultado.
    st.subheader('Descargas y transparencia')  # Mostramos archivos auditados del equipo.
    seleccion=st.multiselect('Estados que se incluirán en el CSV',sorted(tabla.estado.unique()),default=sorted(tabla.estado.unique()))  # Permitimos filtrar exportaciones.
    exportar=tabla.loc[tabla.estado.isin(seleccion),['ID_POLIGONO','estado','municipio','AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']]  # Separamos tabla completa sin variables auxiliares.
    st.download_button('Descargar selección CSV',exportar.to_csv(index=False).encode('utf-8-sig'),'AgroBYTE_predicciones_seleccion.csv','text/csv')  # Descargamos subconjunto georreferenciado.
    minimo=tabla[['ID_POLIGONO','RENDIMIENTO_PREDICHO_T_HA']].copy()  # Preparamos un CSV mínimo para evaluación.
    st.download_button('Descargar 59 predicciones (ID + t/ha)',minimo.to_csv(index=False).encode('utf-8-sig'),'AgroBYTE_59_predicciones_minimo.csv','text/csv')  # Conservamos exactamente 59 filas.
    st.markdown('**Fuentes de información:** bases y dataset FIRA AgroCebada 2026; cálculos y modelos desarrollados por el equipo AgroBYTE.')  # Identificamos procedencia.
    st.markdown('**Código y metodología:** los scripts y documentos acompañan al prototipo en el repositorio de entrega. Consultamos `README.md` y `documentacion/` para reproducir cálculos.')  # Explicamos trazabilidad.
    st.json({'estado':manifiesto['status'],'algoritmo':'ExtraTreesRegressor','variables':manifiesto['caracteristicas'],'n_entrenamiento':138,'n_ocultas':59,'integridad_SHA256':'validada al abrir la aplicación'})  # Hacemos visibles características y auditoría.
    st.warning('Antes de publicar en una URL abierta, confirmamos con FIRA que se permite exponer las coordenadas y polígonos derivados de sus parcelas.')  # Reconocemos responsabilidad de datos.


# ==================== 12. PIE DE PÁGINA ====================
st.divider()  # Separamos producto y créditos.
st.caption('AgroBYTE 3.0 · Prototipo académico independiente para el Reto AgroCebada FIRA 2026 · Rendimiento y escenarios estimados, no garantías de cosecha.')  # Evitamos parecer una plataforma oficial.
