"""Pruebas simples de las rutas portables sin necesidad del dataset oficial."""


# ==================== 1. IMPORTACIÓN DE LIBRERÍAS ====================
import tempfile  # Creamos una carpeta temporal para probar rutas sin tocar el proyecto real.
import unittest  # Utilizamos las pruebas estándar que incluye Python.
from pathlib import Path  # Construimos carpetas de prueba independientes del sistema.
from codigo.rutas import carpeta_dataset, raiz_proyecto, validar_carpeta_salida  # Probamos el módulo usado por todos los scripts.



# ==================== 2. PRUEBAS DE PORTABILIDAD ====================
class PruebasRutas(unittest.TestCase):  # Organizamos las comprobaciones en una clase de pruebas automatizadas.
    def test_dataset_relativo_a_raiz(self):  # Comprobamos que el origen por defecto esté junto al código.
        self.assertEqual(carpeta_dataset(), (raiz_proyecto() / 'DATASET_RETO_AGRO_2026').resolve())  # Verificamos que no haya rutas personales.

    def test_salida_separada(self):  # Comprobamos que un directorio externo al dataset sea válido.
        with tempfile.TemporaryDirectory() as temporal:  # Creamos un espacio temporal que se elimina al terminar.
            datos = Path(temporal) / 'datos'  # Simulamos una carpeta de originales.
            salida = Path(temporal) / 'salidas'  # Simulamos una carpeta independiente de resultados.
            self.assertEqual(validar_carpeta_salida(datos, salida), salida.resolve())  # Aceptamos guardar fuera de los originales.

    def test_prohibir_salida_dentro_de_originales(self):  # Verificamos la protección de datos recibidos.
        with tempfile.TemporaryDirectory() as temporal:  # Aislamos las carpetas de prueba.
            datos = Path(temporal) / 'DATASET_RETO_AGRO_2026'  # Simulamos el dataset sin copiar datos reales.
            with self.assertRaises(ValueError):  # Exigimos un error al intentar guardar dentro del dataset.
                validar_carpeta_salida(datos, datos / 'resultados')  # Comprobamos la restricción de escritura.



# ==================== 3. EJECUCIÓN DE PRUEBAS ====================
if __name__ == '__main__':  # Permitimos ejecutar esta prueba también como script independiente.
    unittest.main()  # Mostramos los resultados de nuestras comprobaciones.
