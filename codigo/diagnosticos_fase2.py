# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
# estudiamos los errores de la fase 2 sin volver a entrenar con las 59 parcelas de evaluación.
from pathlib import Path  # Construimos rutas que funcionan en distintos sistemas operativos.
import argparse  # Leemos opciones de ejecución desde la terminal.
from rutas import carpeta_dataset, raiz_proyecto, validar_carpeta_salida  # Localizamos y protegemos archivos comunes.
import numpy as np  # calculamos promedios y diferencias de predicciones.
import pandas as pd  # unimos predicciones fuera de muestra por parcela.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # calculamos errores verificables.


# ==================== 2. DIAGNÓSTICOS DE LA FASE 2 ====================




# ==================== EJECUCIÓN PRINCIPAL DEL EXPERIMENTO ====================
def main():  # Revisamos que las predicciones experimentales conserven sus identidades y unidades.
    cli = argparse.ArgumentParser(description='Diagnósticos de predicciones AgroBYTE')  # Declaramos las opciones para trabajar en cualquier equipo.
    cli.add_argument('--datos', type=Path, default=None)  # Admitimos otro dataset de prueba sin editar el programa.
    cli.add_argument('--salida', type=Path, default=raiz_proyecto() / 'salidas' / 'fase2')  # Ubicamos las métricas generadas por la Fase 2.
    opciones = cli.parse_args()  # Interpretamos los argumentos recibidos.
    RAIZ_PROYECTO = raiz_proyecto()  # Localizamos la raíz del repositorio sin usar nombres de usuario.
    BASE = carpeta_dataset(opciones.datos)  # Elegimos la carpeta local del dataset o la ruta explícita de pruebas.
    DIR = validar_carpeta_salida(BASE, opciones.salida) / 'resultados'  # Guardamos y leemos diagnósticos solamente en salidas/fase2/resultados.
    filas = pd.read_csv(DIR / 'predicciones_cv_repetida.csv')  # leemos predicciones hechas sin entrenar sobre cada parcela de validación.
    val = pd.read_csv(DIR / 'predicciones_cv_anidada.csv')  # examinamos las predicciones externas de la selección anidada.
    parcelas = pd.read_csv(DIR / 'predicciones_59_EXPERIMENTALES_NO_ENTREGAR.csv')  # validamos solamente estructura y plausibilidad de las 59 salidas.
    coincidencias = sorted(BASE.glob('ID_area_rendimiento*.csv'))  # buscamos el CSV oficial sin depender del sufijo de Windows.
    if len(coincidencias) != 1:  # verificamos que solo exista un archivo de rendimientos original.
        raise FileNotFoundError('No encontramos un único CSV oficial de rendimientos.')  # detenemos el diagnóstico si no podemos identificar la fuente.
    objetivo = pd.read_csv(coincidencias[0])  # leemos rendimientos y etiquetas sin modificar su archivo.
    assert len(parcelas) == 59 and parcelas['ID_POLIGONO'].is_unique  # garantizamos exactamente 59 identificadores distintos.
    assert set(parcelas['ID_POLIGONO']) == set(objetivo.loc[objetivo['CONJUNTO'].eq('PREDICCION'), 'ID_POLIGONO'])  # exigimos igualdad con el conjunto oculto de FIRA.
    assert np.isfinite(parcelas['RENDIMIENTO_PREDICHO_T_HA']).all()  # rechazamos rendimientos indefinidos o infinitos.
    assert (parcelas['RENDIMIENTO_PREDICHO_T_HA'] > 0).all()  # comprobamos que no haya rendimientos físicamente negativos.
    assert np.allclose(parcelas['PRODUCCION_ESTIMADA_T'], parcelas['AREA_HA'] * parcelas['RENDIMIENTO_PREDICHO_T_HA'], atol=0.0005)  # verificamos la fórmula de toneladas totales.
    resumen_estados = []  # preparamos una tabla de error para cada estado en la evaluación anidada.
    for estado, parte in val.groupby('estado'):  # separamos las predicciones según el estado de cada parcela.
        resumen_estados.append({'estado': estado, 'n': len(parte), 'rmse': float(np.sqrt(mean_squared_error(parte['real_t_ha'], parte['predicho_t_ha']))), 'mae': float(mean_absolute_error(parte['real_t_ha'], parte['predicho_t_ha'])), 'sesgo_promedio_pred_menos_real': float(np.mean(parte['error_t_ha']))})  # cuantificamos magnitud y dirección del error.
    pd.DataFrame(resumen_estados).to_csv(DIR / 'errores_cv_anidada_por_estado.csv', index=False)  # documentamos la equidad geográfica del error observado.
    medias = []  # registramos mezclas de modelos solo como experimentos adicionales.
    for seed, base in filas.groupby('semilla'):  # comparamos los mismos folds y parcelas bajo una semilla.
        geo = base.loc[base['candidato'].eq('geo_extratrees'), ['ID_POLIGONO', 'real_t_ha', 'predicho_t_ha']].rename(columns={'predicho_t_ha':'geo'})  # recuperamos el predictor geográfico.
        completo = base.loc[base['candidato'].eq('completo_extratrees'), ['ID_POLIGONO', 'predicho_t_ha']].rename(columns={'predicho_t_ha':'completo'})  # recuperamos el predictor con todas las variables.
        cat = base.loc[base['candidato'].eq('reducido_catboost'), ['ID_POLIGONO', 'predicho_t_ha']].rename(columns={'predicho_t_ha':'catboost'})  # recuperamos el predictor reducido de boosting.
        unidos = geo.merge(completo, on='ID_POLIGONO', validate='one_to_one').merge(cat, on='ID_POLIGONO', validate='one_to_one')  # garantizamos emparejamiento exacto por ID.
        for nombre, valor in [('solo_geografia', unidos['geo']), ('75_geo_25_completo', 0.75*unidos['geo'] + 0.25*unidos['completo']), ('50_geo_50_completo', 0.5*unidos['geo'] + 0.5*unidos['completo']), ('75_geo_25_catboost', 0.75*unidos['geo'] + 0.25*unidos['catboost']), ('50_geo_50_catboost', 0.5*unidos['geo'] + 0.5*unidos['catboost'])]:  # evaluamos ponderaciones fijas fáciles de interpretar.
            medias.append({'semilla': seed, 'modelo': nombre, 'rmse': float(np.sqrt(mean_squared_error(unidos['real_t_ha'], valor))), 'mae': float(mean_absolute_error(unidos['real_t_ha'], valor))})  # calculamos métricas sobre los mismos ejemplos reservados.
    pd.DataFrame(medias).to_csv(DIR / 'mezclas_exploratorias_cv.csv', index=False)  # guardamos los ensambles como hipótesis, nunca como selección independiente.
    print('ESTADOS ANIDADA')  # distinguimos el diagnóstico de errores regionales.
    print(pd.DataFrame(resumen_estados).to_string(index=False))  # mostramos las métricas por región.
    print('MEZCLAS EXPERIMENTALES')  # identificamos las combinaciones que no hemos validado independientemente.
    print(pd.DataFrame(medias).groupby('modelo')[['rmse','mae']].mean().sort_values('rmse').to_string())  # ordenamos hipótesis según su error descriptivo.
    print('RANGO PREDICCIONES FIRA', float(parcelas['RENDIMIENTO_PREDICHO_T_HA'].min()), float(parcelas['RENDIMIENTO_PREDICHO_T_HA'].max()))  # revisamos extremos sin compararlos con etiquetas ocultas.


# ==================== 3. INICIO SEGURO ====================
if __name__ == '__main__':  # Evitamos ejecutar diagnósticos automáticamente cuando importamos este módulo.
    main()  # Ejecutamos los controles a petición del equipo.
