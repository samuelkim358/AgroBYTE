"""Contamos palabras por campo del borrador que pide la plataforma FIRA."""


# ==================== 1. IMPORTACIÓN DE BIBLIOTECAS ====================
from pathlib import Path  # Localizamos el documento dentro de esta carpeta.
import re  # Identificamos palabras y encabezados numerados.


# ==================== 2. DEFINICIÓN DE LOS LÍMITES DE LAS BASES ====================
LIMITES = {1:(400,550),2:(100,200),3:(185,215),4:(600,700),5:(760,820),6:(285,315),7:(380,420)}  # Utilizamos margen para campos sin rango exacto publicado.


# ==================== 3. LECTURA Y CÁLCULO DE EXTENSIONES ====================
contenido = (Path(__file__).resolve().parent/'INFORME_TECNICO_PARA_PLATAFORMA.md').read_text(encoding='utf-8')  # Leemos el borrador completo.
secciones = re.split(r'(?m)^## ([1-7])\. ([^\n]+)\n',contenido)  # Separamos solo apartados oficiales numerados.
for i in range(1,len(secciones),3):  # Examinamos cada campo del formulario FIRA.
    indice = int(secciones[i])  # Recuperamos el número de apartado.
    titulo = secciones[i+1]  # Recuperamos el nombre oficial.
    cuerpo = secciones[i+2].split('## ')[0]  # Excluimos campos posteriores.
    palabras = len(re.findall(r'\b[\wáéíóúñü/-]+\b',cuerpo))  # Calculamos longitud orientativa.
    inferior,superior = LIMITES[indice]  # Consultamos los límites de revisión interna.
    print(f'{indice}. {titulo}: {palabras} palabras (orientación {inferior}–{superior})')  # Comunicamos el tamaño al equipo.
print('IMPORTANTE: el contador de la plataforma puede tratar guiones y números de forma distinta.')  # Pedimos verificar el formulario real.
