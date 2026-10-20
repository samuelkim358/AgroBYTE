"""Probamos las distancias y los controles de integridad sin necesitar el dataset de FIRA."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import unittest  # Creamos pruebas automáticas estándar para compartirlas con todo el equipo.
import numpy as np  # Preparamos coordenadas y distancias de prueba.
import pandas as pd  # Construimos tablas pequeñas para comprobar errores de entrada.
from codigo.auditoria_geografica_final import calcular_cobertura, distancias_haversine_km  # Recuperamos solamente las funciones que vamos a validar.



# ==================== 2. VERIFICACIÓN DE DISTANCIAS ====================
class TestDistancias(unittest.TestCase):  # Agrupamos los experimentos reproducibles de la fórmula de Haversine.
    def test_misma_ubicacion_distancia_cero(self):  # Comprobamos que un punto tenga distancia cero respecto a sí mismo.
        distancia = distancias_haversine_km(np.array([[19.0, -98.0]]), np.array([[19.0, -98.0]]))  # Calculamos el caso básico sin separación.
        self.assertAlmostEqual(float(distancia[0, 0]), 0.0, places=9)  # Exigimos una coincidencia numérica cercana a cero.

    def test_un_grado_longitud_ecuador(self):  # Contrastamos una distancia de referencia conocida en el ecuador.
        distancia = distancias_haversine_km(np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]]))  # Calculamos aproximadamente un grado de longitud.
        self.assertAlmostEqual(float(distancia[0, 0]), 111.195, places=2)  # Verificamos que la conversión a kilómetros sea coherente.

    def test_matriz_de_distancias(self):  # Confirmamos que la función compare todas las parcelas con todos los vecinos.
        distancia = distancias_haversine_km(np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 0.0]]))  # Construimos una matriz de dos filas y una columna.
        self.assertEqual(distancia.shape, (2, 1))  # Comprobamos que no perdamos ninguna pareja de puntos.



# ==================== 3. COMPROBACIÓN DEL CSV CANDIDATO ====================
class TestIntegridad(unittest.TestCase):  # Verificamos que no se acepten archivos de predicción incompletos.
    def test_rechaza_predicciones_duplicadas(self):  # Comprobamos que dos resultados con el mismo identificador se rechacen.
        tabla = pd.DataFrame({'ID_POLIGONO': ['AGC_001', 'AGC_001']})  # Simulamos una tabla incorrecta de dos identificadores repetidos.
        with self.assertRaises(ValueError):  # Esperamos un mensaje de error antes de iniciar cualquier cálculo espacial.
            calcular_cobertura(pd.DataFrame(), tabla)  # Probamos que el control detecte una entrega duplicada.

    def test_rechaza_predicciones_incompletas(self):  # Comprobamos que se rechace una entrega con menos de 59 parcelas.
        tabla = pd.DataFrame({'ID_POLIGONO': ['AGC_001']})  # Simulamos una tabla que solo incluye una parcela.
        with self.assertRaises(ValueError):  # Exigimos una excepción de integridad explícita.
            calcular_cobertura(pd.DataFrame(), tabla)  # Confirmamos que la auditoría no interprete muestras incompletas.


if __name__ == '__main__':  # Ejecutamos las pruebas únicamente cuando solicitamos este archivo.
    unittest.main()  # Activamos el verificador estándar de Python.
