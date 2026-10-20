"""Comprobamos la lógica de índices, clima y separación de etiquetas de AgroBYTE."""


# ==================== IMPORTACIÓN DE LIBRERÍAS ====================
import sys  # Habilitamos la importación del módulo auxiliar desde la carpeta tests.
import unittest  # Ejecutamos comprobaciones automáticas sin herramientas externas.
from pathlib import Path  # Ubicamos el módulo del proyecto de manera portable.
import pandas as pd  # Creamos datos pequeños de prueba sin descargar el dataset oficial.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codigo'))  # Añadimos el directorio de los módulos científicos.
from fase5_indices_clima import grupos_de_indices, grupos_clima, familias_columnas, configuraciones  # Probamos reglas científicas sin entrenar.


# ==================== PRUEBAS DE INTEGRIDAD DE CARACTERÍSTICAS ====================
class PruebasFase5(unittest.TestCase):  # Agrupamos controles para su ejecución con unittest.
    def test_indices_agrupados_por_sensor_y_nombre(self):  # Comprobamos que medimos índices por familias coherentes.
        cols = ['Sentinel_2_ndvi_promedio_mes_04','Sentinel_2_ndvi_promedio_ciclo_maximo','Planet_ndvi_promedio_mes_04','Planet_msavi_promedio_mes_05','Sentinel_2_n_meses_observados']  # Simulamos dos sensores y control de calidad.
        indices,sensores = grupos_de_indices(cols)  # Reconstruimos grupos sin etiquetas reales.
        self.assertEqual(len(indices['NDVI']),3)  # Conservamos NDVI de ambos sensores bajo su familia.
        self.assertEqual(len(sensores['Sentinel_2:NDVI']),2)  # Separamos las observaciones de Sentinel-2.
        self.assertNotIn('N',indices)  # Evitamos mezclar recuentos de observaciones con índices.
    def test_tres_variables_climaticas(self):  # Comprobamos que clima se clasifica sin mezclar temperatura y lluvia.
        clima = grupos_clima(['prec_mm_2025_mes_04','tmin_c_2025_mes_04','tmax_c_2025_mes_04'])  # Simulamos un mes climático.
        self.assertTrue(all(len(x)==1 for x in clima.values()))  # Exigimos una columna por variable.
    def test_geografia_conserva_cinco_columnas(self):  # Conservamos exactamente la arquitectura del modelo principal.
        tabla = pd.DataFrame(columns=['ID_POLIGONO','CONJUNTO','RENDIMIENTO_T_HA','AREA_HA','latitud','longitud','estado','municipio','Sentinel_2_ndvi_promedio_mes_04','prec_mm_2025_mes_04','pendiente_grados','Sentinel_2_n_meses_observados'])  # Construimos un esquema mínimo consistente.
        grupos = familias_columnas(tabla)  # Clasificamos las columnas del experimento.
        self.assertEqual(configuraciones(grupos)['geo'], ['AREA_HA','latitud','longitud','estado','municipio'])  # Verificamos el grupo exacto de la Fase 4.
        self.assertEqual(len(configuraciones(grupos)['completo']),len(tabla.columns)-3)  # Impedimos excluir silenciosamente características.


if __name__=='__main__':  # Permitimos correr las pruebas como script independiente.
    unittest.main()  # Verificamos los invariantes científicos del módulo.
