"""Verificamos las reglas de integridad del CSV candidato sin entrenar modelos."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import sys  # Facilitamos importar los archivos existentes desde la carpeta de código.
import unittest  # Escribimos pruebas automáticas compatibles con el proyecto.
from pathlib import Path  # Resolvemos rutas independientes del sistema operativo.
import pandas as pd  # Creamos ejemplos mínimos de tablas para nuestras pruebas.



# ==================== 2. MÓDULO DE VALIDACIÓN ====================
RAIZ = Path(__file__).resolve().parent.parent  # Identificamos la raíz del proyecto que contiene los tests.
sys.path.insert(0, str(RAIZ / 'codigo'))  # Incorporamos los scripts compartidos a la ruta de importación.
from fase4_cierre_modelo import comprobar_predicciones  # Probamos el mismo validador que se utiliza en la exportación.



# ==================== 3. CASOS DE PRUEBA ====================
class ValidacionCandidato(unittest.TestCase):  # Agrupamos verificaciones del archivo final para el dashboard.
    def crear_ejemplo(self):  # Preparamos 59 registros sintéticos exclusivamente para probar la estructura.
        tabla = pd.DataFrame({'ID_POLIGONO': [f'PRUEBA_{i:03d}' for i in range(59)], 'AREA_HA': [2.0] * 59, 'RENDIMIENTO_PREDICHO_T_HA': [3.5] * 59})  # Creamos IDs únicos y rendimientos positivos.
        tabla['PRODUCCION_ESTIMADA_T'] = tabla['AREA_HA'] * tabla['RENDIMIENTO_PREDICHO_T_HA']  # Aplicamos la fórmula de toneladas totales.
        return tabla  # Devolvemos un ejemplo sin utilizar los rendimientos oficiales de FIRA.

    def test_acepta_59_parcelas_correctas(self):  # Confirmamos que el validador acepta tablas consistentes.
        tabla = self.crear_ejemplo()  # Generamos los datos sintéticos del caso válido.
        comprobar_predicciones(tabla, set(tabla['ID_POLIGONO']))  # Exigimos que la comprobación termine sin errores.

    def test_rechaza_identificadores_duplicados(self):  # Evitamos publicar predicciones para la misma parcela dos veces.
        tabla = self.crear_ejemplo()  # Construimos primero el caso correcto.
        tabla.loc[1, 'ID_POLIGONO'] = tabla.loc[0, 'ID_POLIGONO']  # Introducimos intencionalmente un duplicado.
        with self.assertRaises(ValueError):  # Comprobamos que el programa detecta la inconsistencia.
            comprobar_predicciones(tabla, set(tabla['ID_POLIGONO']))  # Ejecutamos la comprobación de IDs.

    def test_rechaza_produccion_incorrecta(self):  # Evitamos errores de unidades y multiplicación en el dashboard.
        tabla = self.crear_ejemplo()  # Construimos un ejemplo con cifras correctas.
        tabla.loc[0, 'PRODUCCION_ESTIMADA_T'] = 999.0  # Simulamos una producción total calculada incorrectamente.
        with self.assertRaises(ValueError):  # Comprobamos que el programa detecta la discrepancia.
            comprobar_predicciones(tabla, set(tabla['ID_POLIGONO']))  # Validamos las toneladas calculadas.


if __name__ == '__main__':  # Permitimos ejecutar las pruebas desde la terminal si se solicita.
    unittest.main()  # Ejecutamos todos los casos de validación definidos.
