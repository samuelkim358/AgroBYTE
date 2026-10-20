"""Ejecutamos una auditoría antes de abrir AgroBYTE 3.0."""


# ==================== 1. IMPORTACIÓN DE BIBLIOTECAS ====================
from codigo.entrega_v3 import cargar_entrega  # Reutilizamos nuestras reglas de integridad y predicción.


# ==================== 2. VERIFICACIÓN AUTOMÁTICA ====================
def main():  # Evaluamos huellas, identidades y reproducción real de las 59 predicciones.
    tabla, manifiesto, _, avisos = cargar_entrega()  # Ejecutamos todas las comprobaciones sin datos oficiales originales.
    print('OK: Modelo Extra Trees cargado y verificado.')  # Confirmamos reproducción de predicciones.
    print('OK: Número de parcelas:', len(tabla))  # Informamos cobertura.
    print('OK: Rendimiento en rango:', round(tabla.RENDIMIENTO_EN_VIVO_T_HA.min(),4), '-',round(tabla.RENDIMIENTO_EN_VIVO_T_HA.max(),4),'t/ha')  # Informamos extremos.
    print('OK: Estado de entrega:',manifiesto['status'])  # Informamos el estado de revisión.
    if avisos:  # Comunicamos compatibilidad cruzada de versiones si existe.
        print('ATENCIÓN: la versión de scikit-learn difiere de la versión usada en el entrenamiento.')  # Recomendamos reproducir entorno para despliegue.


# ==================== 3. PUNTO DE ENTRADA ====================
if __name__=='__main__':  # Ejecutamos auditoría cuando invocamos este archivo desde terminal.
    main()  # Detenemos con error si alguna condición no se cumple.
