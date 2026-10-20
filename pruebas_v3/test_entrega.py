"""Probamos que la entrega FIRA reproduce inferencias y cálculos documentados."""


# ==================== 1. IMPORTACIÓN DE BIBLIOTECAS ====================
import copy  # Construimos alteraciones controladas sin modificar el manifiesto.
import json  # Comprobamos la estructura del mapa generado.
import unittest  # Organizamos pruebas automáticas de Python.
from pathlib import Path  # Localizamos los recursos del paquete.
import numpy as np  # Contrastamos valores numéricos con tolerancias estrictas.
import pandas as pd  # Construimos casos de prueba de entradas y salidas.
from codigo.entrega_v3 import RAIZ, cargar_entrega, predecir_parcela, simular, validar_tablas, verificar_huellas  # Recuperamos las funciones que protege nuestro producto.


# ==================== 2. PRUEBAS DE DATOS Y MODELO ====================
class PruebasEntrega(unittest.TestCase):  # Separamos la validación de la interfaz para ejecutarla sin navegador.
    @classmethod
    def setUpClass(cls):  # Cargamos una sola vez los árboles entrenados.
        cls.tabla,cls.manifiesto,cls.modelo,_=cargar_entrega()  # Reproducimos la comprobación principal de SHA-256 e inferencia.

    def test_01_cobertura_59(self):  # Verificamos tamaño y claves unívocas.
        self.assertEqual(len(self.tabla),59)  # Exigimos todas las parcelas ocultas.
        self.assertEqual(self.tabla.ID_POLIGONO.nunique(),59)  # Rechazamos ID repetidos.

    def test_02_huellas(self):  # Confirmamos que el paquete distribuido está firmado.
        verificar_huellas(self.manifiesto)  # Ejecutamos la verificación completa.

    def test_03_inferencia_real_coincide(self):  # Ejecutamos el modelo en lugar de leer el CSV.
        for _,fila in self.tabla.iterrows():  # Comprobamos cada terreno en su registro original.
            y,ton=predecir_parcela(self.modelo,self.tabla,fila.ID_POLIGONO)  # Regeneramos rendimiento y toneladas.
            self.assertAlmostEqual(y,float(fila.RENDIMIENTO_PREDICHO_T_HA),places=6)  # Comprobamos precisión del rendimiento.
            self.assertAlmostEqual(ton,float(fila.PRODUCCION_ESTIMADA_T),places=5)  # Comprobamos producción total.

    def test_04_rechazamos_id_desconocido(self):  # Limitamos las predicciones a las parcelas del concurso.
        with self.assertRaises(ValueError):  # Esperamos rechazo explícito.
            predecir_parcela(self.modelo,self.tabla,'PARCELA_DESCONOCIDA')  # No permitimos inferencia fuera del alcance.

    def test_05_rechazamos_prediccion_duplicada(self):  # Detectamos sustituciones accidentales de registros.
        csv=self.tabla[['ID_POLIGONO','estado','municipio','AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']].copy()  # Construimos una tabla artificial.
        entradas=self.tabla[['ID_POLIGONO','AREA_HA_entrada','latitud','longitud','estado_entrada','municipio_entrada']].copy()  # Extraemos entradas.
        entradas.columns=['ID_POLIGONO','AREA_HA','latitud','longitud','estado','municipio']  # Restauramos nombres originales.
        csv.loc[0,'ID_POLIGONO']=csv.loc[1,'ID_POLIGONO']  # Introducimos un duplicado deliberado.
        with self.assertRaises(ValueError):  # Esperamos error antes de calcular cifras.
            validar_tablas(csv,entradas)  # Ejecutamos la validación de negocio.

    def test_06_cobertura_poligonos(self):  # Aseguramos que el visor geográfico no muestra fincas extrañas.
        ruta=RAIZ/'datos'/'poligonos_59.geojson'  # Recuperamos las geometrías derivadas del reto.
        datos=json.loads(ruta.read_text(encoding='utf-8'))  # Leemos geometrías en GeoJSON.
        ids={f['properties']['ID_POLIGONO'] for f in datos['features']}  # Recuperamos identificadores visibles.
        self.assertEqual(ids,set(self.tabla.ID_POLIGONO))  # Exigimos 59 polígonos equivalentes.

    def test_07_rechazamos_produccion_inconsistente(self):  # Controlamos un error frecuente de superficie por rendimiento.
        csv=self.tabla[['ID_POLIGONO','estado','municipio','AREA_HA','RENDIMIENTO_PREDICHO_T_HA','PRODUCCION_ESTIMADA_T']].copy()  # Creamos copia sin tocar el archivo.
        entradas=self.tabla[['ID_POLIGONO','AREA_HA_entrada','latitud','longitud','estado_entrada','municipio_entrada']].copy()  # Construimos entradas.
        entradas.columns=['ID_POLIGONO','AREA_HA','latitud','longitud','estado','municipio']  # Recuperamos nombres de interfaz.
        csv.loc[0,'PRODUCCION_ESTIMADA_T']+=1.0  # Alteramos intencionalmente una tonelada.
        with self.assertRaises(ValueError):  # Esperamos rechazar la salida.
            validar_tablas(csv,entradas)  # Verificamos consistencia del cálculo.


# ==================== 3. PRUEBAS DE SIMULADOR ECONÓMICO ====================
class PruebasSimulador(unittest.TestCase):  # Comprobamos casos normales y condiciones extremas.
    def test_08_produccion_ingreso_y_costos(self):  # Calculamos un ejemplo con valores controlados.
        s=simular(2,4,5000,1000,2000,500,500,10000,12,6)  # Elegimos dos hectáreas y seis meses de crédito simple.
        self.assertAlmostEqual(s['produccion_t'],8)  # Validamos toneladas totales.
        self.assertAlmostEqual(s['ingreso_mxn'],40000)  # Validamos ventas brutas.
        self.assertAlmostEqual(s['costo_mxn'],8000)  # Validamos suma de costos.
        self.assertAlmostEqual(s['interes_simple_mxn'],600)  # Validamos interés simple semestral.
        self.assertAlmostEqual(s['margen_estimado_mxn'],31400)  # Validamos margen del ejemplo.

    def test_09_rechazamos_negativos(self):  # Detectamos entradas incompatibles con el simulador.
        with self.assertRaises(ValueError):  # Esperamos validación restrictiva.
            simular(2,4,-1,0,0,0,0)  # Rechazamos precio negativo.

    def test_10_sin_precio_no_inventamos_equilibrio(self):  # Evitamos divisiones por cero.
        self.assertIsNone(simular(3,4,0,0,0,0,0)['rendimiento_equilibrio_t_ha'])  # Reportamos ausencia de punto de equilibrio.


# ==================== 4. INICIO DE PRUEBAS ====================
if __name__ == '__main__':  # Permitimos ejecutar esta suite independientemente.
    unittest.main()  # Informamos aprobaciones o fallas.
