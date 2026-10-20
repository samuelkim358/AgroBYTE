"""Comprobaciones rápidas del módulo de XGBoost sin utilizar los originales de FIRA."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import unittest  # Ejecutamos pruebas con herramientas incluidas en Python.
import numpy as np  # Construimos rendimientos ficticios para probar los algoritmos.
import pandas as pd  # Construimos pequeñas tablas de prueba con el formato esperado.
from codigo.fase3_xgboost import calcular_metricas, construir_candidato, predecir_fuera_de_muestra  # Probamos las funciones públicas de la nueva fase.
from sklearn.model_selection import StratifiedKFold  # Preparamos particiones equivalentes a las usadas en el experimento.


# ==================== 2. CASOS DE PRUEBA AUTOMATIZADOS ====================
class TestFase3(unittest.TestCase):  # Agrupamos pruebas independientes de los datos originales.
    def test_metricas_perfectas(self):  # Comprobamos los valores ideales cuando no existe error de predicción.
        resultados = calcular_metricas([2.0, 3.0, 4.0], [2.0, 3.0, 4.0])  # Evaluamos tres predicciones exactas.
        self.assertAlmostEqual(resultados['rmse'], 0.0)  # Verificamos que RMSE sea cero.
        self.assertAlmostEqual(resultados['mae'], 0.0)  # Verificamos que MAE sea cero.
        self.assertAlmostEqual(resultados['r2'], 1.0)  # Verificamos que R² sea uno.

    def test_xgboost_pipeline_numerico_y_categorico(self):  # Verificamos la integración con el preprocesamiento usado en la Fase 2.
        tabla = pd.DataFrame({'AREA_HA': [1,2,3,4,5,6,7,8], 'latitud': [19.0,19.1,19.2,19.3,19.4,19.5,19.6,19.7], 'longitud': [-98.0,-98.1,-98.2,-98.3,-98.4,-98.5,-98.6,-98.7], 'estado': ['Puebla']*4+['Hidalgo']*4, 'municipio': ['A']*4+['B']*4})  # Simulamos parcelas con números y categorías.
        rendimiento = np.array([3.0,3.3,3.5,4.0,2.7,3.1,3.8,4.2])  # Definimos únicamente valores ficticios de entrenamiento.
        modelo = construir_candidato('xgb_conservador', tabla)  # Creamos el modelo con el mismo pipeline categórico del proyecto.
        modelo.fit(tabla, rendimiento)  # Ajustamos todas las transformaciones dentro del estimador.
        estimadas = modelo.predict(tabla.iloc[:3])  # Comprobamos que se pueden generar tres predicciones.
        self.assertEqual(len(estimadas), 3)  # Confirmamos el tamaño correcto de la salida.
        self.assertTrue(np.isfinite(estimadas).all())  # Rechazamos valores faltantes o infinitos.

    def test_validacion_cubre_cada_parcela_una_vez(self):  # Confirmamos que los pliegues de prueba no pierden ni duplican parcelas.
        n = 30  # Definimos una muestra pequeña independiente del concurso.
        tabla = pd.DataFrame({'ID_POLIGONO': [f'TEST_{i:03d}' for i in range(n)], 'estado': ['Puebla']*15+['Hidalgo']*15, 'municipio': ['A']*15+['B']*15, 'AREA_HA': np.arange(1, n+1), 'latitud': np.linspace(19.1,19.7,n), 'longitud': np.linspace(-98.1,-98.7,n), 'RENDIMIENTO_T_HA': np.linspace(2.0,4.5,n)})  # Construimos valores de prueba con los campos mínimos geográficos.
        cortes = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(tabla, tabla['estado']))  # Construimos cinco particiones con ambos estados representados.
        metricas, detalle = predecir_fuera_de_muestra(tabla, ('geo_extratrees', 'geografia', 'extratrees'), cortes, 42, 'parcelas')  # Ejecutamos un modelo de control sin fuga de etiquetas.
        self.assertEqual(len(detalle), n)  # Confirmamos una predicción por ejemplo.
        self.assertTrue(detalle['ID_POLIGONO'].is_unique)  # Impedimos duplicados en el resultado.
        self.assertTrue(np.isfinite(detalle['predicho_t_ha']).all())  # Exigimos predicciones finitas.
        self.assertTrue(np.isfinite(metricas['rmse']))  # Exigimos que el error se pueda calcular.


# ==================== 3. EJECUCIÓN OPCIONAL DE LAS PRUEBAS ====================
if __name__ == '__main__':  # Permitimos iniciar estas pruebas manualmente desde la terminal.
    unittest.main()  # Ejecutamos todos los casos de este archivo.
