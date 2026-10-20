"""Probamos la validación del dashboard sin necesitar el dataset ni iniciar Streamlit."""


# ==================== 1. IMPORTACIÓN DE PRUEBAS ====================
import unittest  # Creamos pruebas reproducibles mediante la biblioteca estándar.
import numpy as np  # Generamos ejemplos numéricos para verificar tolerancias.
import pandas as pd  # Construimos tablas de ejemplo sin datos reales del concurso.
from codigo.datos_dashboard import validar_predicciones, resumen_indicadores, csv_exportacion  # Probamos la lógica pura separada de la interfaz.


# ==================== 2. TABLA SINTÉTICA PARA PRUEBAS ====================
def tabla_ejemplo():  # Construimos 59 parcelas ficticias para detectar errores del código.
    return pd.DataFrame({'ID_POLIGONO': [f'PRUEBA_{i:03d}' for i in range(59)], 'estado': ['Estado de prueba'] * 59, 'municipio': ['Municipio de prueba'] * 59, 'AREA_HA': [2.0] * 59, 'RENDIMIENTO_PREDICHO_T_HA': [4.0] * 59, 'PRODUCCION_ESTIMADA_T': [8.0] * 59})  # Evitamos incorporar parcelas de FIRA en tests.


# ==================== 3. CONTROLES DE CALIDAD ====================
class TestDashboardV2(unittest.TestCase):  # Agrupamos verificaciones de las reglas de negocio.
    def test_59_validas(self):  # Comprobamos que una tabla coherente pasa los controles.
        self.assertEqual(len(validar_predicciones(tabla_ejemplo())), 59)  # Verificamos exactamente 59 filas.

    def test_faltan_filas(self):  # Detectamos una exportación incompleta.
        with self.assertRaises(ValueError):  # Exigimos error cuando falta un ID.
            validar_predicciones(tabla_ejemplo().iloc[:-1])  # Simulamos una tabla de 58 parcelas.

    def test_repeticion_id(self):  # Detectamos parcelas repetidas.
        tabla = tabla_ejemplo()  # Preparamos una tabla inicialmente correcta.
        tabla.loc[1, 'ID_POLIGONO'] = tabla.loc[0, 'ID_POLIGONO']  # Duplicamos intencionalmente un ID.
        with self.assertRaises(ValueError):  # Comprobamos que no se acepte.
            validar_predicciones(tabla)  # Validamos la fuente alterada.

    def test_unidades_mal_calculadas(self):  # Detectamos producción inconsistente con hectáreas por rendimiento.
        tabla = tabla_ejemplo()  # Preparamos un ejemplo conocido.
        tabla.loc[0, 'PRODUCCION_ESTIMADA_T'] = 1000.0  # Introducimos un error de unidades.
        with self.assertRaises(ValueError):  # Exigimos rechazo antes de mostrar el dato.
            validar_predicciones(tabla)  # Comprobamos la multiplicación.

    def test_negativos_y_nulos(self):  # Detectamos rendimientos no físicos.
        tabla = tabla_ejemplo()  # Preparamos una fuente correcta.
        tabla.loc[3, 'RENDIMIENTO_PREDICHO_T_HA'] = np.nan  # Introducimos un valor faltante.
        with self.assertRaises(ValueError):  # Impedimos que la app calcule con NaN.
            validar_predicciones(tabla)  # Comprobamos la validación numérica.

    def test_ids_oficiales(self):  # Comprobamos el emparejamiento con una lista esperada.
        tabla = tabla_ejemplo()  # Preparamos una muestra sintética.
        esperados = set(tabla['ID_POLIGONO'])  # Construimos la referencia de identificadores.
        self.assertEqual(len(validar_predicciones(tabla, esperados)), 59)  # Confirmamos coincidencia completa.
        with self.assertRaises(ValueError):  # Detectamos identificadores diferentes.
            validar_predicciones(tabla, {'ID_INEXISTENTE'})  # Simulamos referencia incorrecta.

    def test_resumen_ponderado(self):  # Verificamos que el promedio use superficie, no el promedio simple.
        tabla = tabla_ejemplo().iloc[:2].copy()  # Seleccionamos dos terrenos de prueba.
        tabla.loc[0, ['AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']] = [1.0, 2.0, 2.0]  # Construimos la primera observación.
        tabla.loc[1, ['AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']] = [3.0, 4.0, 12.0]  # Construimos la segunda observación.
        resumen = resumen_indicadores(tabla)  # Recalculamos las cifras del dashboard.
        self.assertAlmostEqual(resumen['rendimiento_ponderado_t_ha'], 3.5)  # Exigimos catorce toneladas divididas entre cuatro hectáreas.

    def test_csv_minimo(self):  # Comprobamos que la descarga mínima solo incluya dos columnas.
        columnas = pd.read_csv(__import__('io').BytesIO(csv_exportacion(tabla_ejemplo(), minimo=True))).columns.tolist()  # Volvemos a leer la exportación.
        self.assertEqual(columnas, ['ID_POLIGONO', 'RENDIMIENTO_PREDICHO_T_HA'])  # Verificamos el contrato de entrega mínima.


# ==================== 4. EJECUCIÓN DE PRUEBAS ====================
if __name__ == '__main__':  # Permitimos ejecutar el archivo de pruebas individualmente.
    unittest.main()  # Iniciamos las comprobaciones cuando el módulo es invocado directamente.
