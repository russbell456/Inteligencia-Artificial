# Fuentes originales

Los archivos fuente ENAHO (CSV) y ESCALE (DBF) permanecen en la raíz del proyecto para conservar intacta la ubicación en que fueron entregados. `src/pipeline.py` los lee desde allí; los productos derivados se escriben en `data/processed/`.

## ESCALE 2025: cobertura y cruce

El `Padron.dbf` disponible contiene identificadores institucionales (`COD_MOD`, `CODLOCAL`), ubicación (`CODGEO`), conteos del Censo Educativo (`TALUM`, `TDOC`, `TSEC`) y coordenadas de servicio educativo. `Padlocal.dbf` aporta localización y geografía, pero no coordenadas numéricas. Se descargó el archivo oficial `01_Matricula_01_2025.zip` (actualización del catálogo: 23/01/2026) desde la ruta alternativa oficial `https://escale.minedu.gob.pe/documents/inicio/Bases%20de%20datos/A.%20Censo%20Educativo/2025/01_Matricula_01.zip`. El ZIP íntegro contiene `Matricula_01.dbf` (178 425 filas; SHA-256 `6D732736D0F71E1488C5797ED3ADDBCA7820B5E7C15CA2AE1255948D3C881FA1`); la copia extraída está en `Matricula_01_2025.dbf`. El diccionario oficial denomina a `COD_MOD` “Id del servicio educativo”.

El pipeline filtra C201 de 3AP (primaria) y 3AS (secundaria), agrega el desglose de matrícula por `COD_MOD + ANEXO`, enlaza con el padrón por la misma llave y suma los registros enlazados por UBIGEO para integrarlos a ENAHO. La auditoría resultante reporta el número de servicios enlazados y sin coincidencia. ENAHO 2025 no incluye `COD_MOD` ni `CODLOCAL` en los módulos personales/educativos disponibles; por ello no existe un cruce persona-escuela demostrable. Los indicadores son contexto distrital y no se atribuye a cada niño la matrícula de una escuela concreta.

Referencias: [catálogo oficial de operaciones ESCALE 2025](https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985), [diccionario del Censo Educativo 2025](https://escale.minedu.gob.pe/documents/10156/10632985/00_Diccionario%2Bde%2Bdatos_CE_2025.pdf), [padrón de instituciones ESCALE](https://escale.minedu.gob.pe/web/inicio/padron-de-iiee).

## ENAHO 2025: diccionario de datos

Se descargó el diccionario oficial `Diccionario_ENAHO_2025.pdf` desde INEI (edición mayo de 2026, documentación de los microdatos ENAHO 2025). Se verificaron en el documento las etiquetas y códigos de P306, P307, P308A, P314A, P1121, P1144 y FACTOR07. P314A usa 1=Sí y 2=No para el uso de Internet el mes anterior; P1144 marca conexión a Internet fija/móvil del hogar (0=no, 1=sí); P1121 es alumbrado eléctrico del hogar (0=no, 1=sí), no conectividad. El catálogo oficial del INEI enlaza los recursos del año 2025.

Fuente: [Diccionario ENAHO 2025, INEI](https://proyectos.inei.gob.pe/iinei/srienaho/Descarga/DocumentosMetodologicos/2025-55/20_Diccionario_2025.pdf); [catálogo de ENAHO 2025 en Datos Abiertos](https://www.datosabiertos.gob.pe/dataset/encuesta-nacional-de-hogares-enaho-2025-instituto-nacional-de-estad%C3%ADstica-e-inform%C3%A1tica-%E2%80%93).
