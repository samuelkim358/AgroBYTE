# AgroBYTE 3.0.1 — Corrección de interfaz

## Incidentes reproducidos

1. `NameError: elegida_global no está definido` al entrar en «Ejecutar modelo» y «Simulador económico». La asignación escribía `eligida_global` y los consumidores `elegida_global`.
2. Mapa con mosaicos **API KEY REQUIRED** de CARTO: fallo de teselas externas, no de nuestras geometrías.

## Cambios

- Unificamos el nombre `elegida_global` y reutilizamos el ID seleccionado entre pestañas.
- Sustituimos el fondo CARTO por **OpenStreetMap**, sin token de API, con opción **Solo parcelas (sin conexión)**.
- Afinamos el contraste del menú lateral y la barra superior cuando se activa el modo oscuro.
- Conservamos el modelo y el conjunto de 59 resultados sin reentrenar.

## Pruebas automáticas

- 25 pruebas de la base histórica.
- 10 pruebas de la versión 3.0.
- 3 pruebas nuevas de navegación y mapas (recorren las seis pantallas con un simulador de componentes Streamlit y renderización real de Folium).
- **38 pruebas superadas** en el entorno de comprobación; la prueba visual final en Windows/Chrome sigue pendiente.

## Comprobación para el equipo

1. Iniciamos `python -m streamlit run app.py`.
2. Abrimos directamente «Ejecutar modelo», seleccionamos una parcela y pulsamos calcular: el resultado debe coincidir con la ficha y el CSV.
3. Abrimos «Simulador económico» antes o después de visitar el mapa; la ficha y calculadora deben funcionar.
4. Abrimos «Mapa y parcelas» y verificamos los polígonos. Probamos alternar **OpenStreetMap** / **Solo parcelas** sin errores.
5. Cambiamos entre modo claro y oscuro.
6. Ejecutamos `python verificar_entrega.py` y `python -m unittest discover -s pruebas_v3 -v`.

El mapa con fondo depende de internet y de las políticas de uso de OpenStreetMap. La opción sin fondo no usa conexión ni API.
