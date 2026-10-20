# Fuentes externas investigadas — propuestas, NO incorporadas aún al modelo

## 1. DGSIAP/SIAP — Rendimiento municipal de cebada 2022–2024

Página oficial: https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php

La estadística agrícola oficial contiene cierre municipal por año y permite consultar rendimientos, producción, superficie, modalidad hídrica y precios por cultivo, con datos disponibles para 2022, 2023 y 2024.

**Experimento propuesto:** filtrar cultivo cebada grano, normalizar entidad y municipio, agregar mediana/promedio de rendimiento municipal 2022–2024 y unir la misma característica a las 197 parcelas. La variable histórica debe representar información disponible antes del cierre de la campaña que se intenta pronosticar. No introducir un rendimiento municipal de 2025 sin demostrar que no incorpora las mismas cosechas que estamos prediciendo.

**Riesgos:** diferencias de variedad/cultivo, agricultura de temporal vs riego, municipios sin registro o pocos productores. La media municipal no representa el rendimiento individual de cada parcela.

## 2. ISRIC SoilGrids — propiedades de suelo

Documentación: https://docs.isric.org/globaldata/soilgrids/

Considerar pH, carbono orgánico, textura y densidad aparente en distintas profundidades. ISRIC informa que su REST API v2 está temporalmente suspendida; para trabajo reproducible sugiere WCS, WebDAV o acceso a capas geográficas por otros medios. No incluir valores de suelo inventados ni atribuir al polígono resolución que no tiene la fuente.

**Prueba posterior:** comprobar cobertura y resolución para las 197 parcelas, convertir unidades conforme metadatos originales y comparar el cambio en RMSE por parcela y municipio.

**Decisión:** por ahora ninguna de estas fuentes entra en las predicciones experimentales. Se evaluarán solo si su integración es verificable y aporta señal consistente en validación.
