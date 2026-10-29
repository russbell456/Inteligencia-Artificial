# Defensa: qué puede y qué no puede enlazarse con ESCALE

## Guion oral (30–45 segundos)

“Descargamos e integramos el archivo oficial de matrícula del Censo Educativo ESCALE 2025. El diccionario indica que C201, cédulas 3AP y 3AS, contiene matrícula por grado, sexo y turno; `COD_MOD` identifica al servicio y `ANEXO` al anexo. Cruzamos matrícula y padrón por `COD_MOD + ANEXO`; los 54 038 servicios encontrados en matrícula tuvieron coincidencia en el padrón. Como ENAHO no trae código modular ni código de local, agregamos matrícula de primaria y secundaria a nivel de UBIGEO para incorporarla como contexto distrital. No afirmamos que sea la escuela del niño: para eso haría falta una llave de institución en ENAHO.”

## Evidencia de la integración

| Fuente | Llave o campos relevantes | Nivel de observación | Uso en este proyecto |
|---|---|---|---|
| ENAHO 2025, módulos disponibles | Llaves de hogar/persona y UBIGEO; no se encontraron `COD_MOD` ni `CODLOCAL` en el módulo educativo/personal | Persona/hogar | Permite cruzar ENAHO entre sí y agrupar ubicación hasta distrito |
| ESCALE `Padron.dbf` | `COD_MOD`, `CODLOCAL`, `CODGEO`, `TALUM`, `TDOC`, `TSEC`, latitud/longitud | Servicio educativo/local | Resume oferta escolar por distrito (UBIGEO) |
| ESCALE Censo Educativo 2025, `Matricula_01.dbf` | `COD_MOD`, `ANEXO`, `NROCED`, `CUADRO`, `D01`–`D12` | Matrícula del servicio por grado, sexo y turno | C201/3AP y C201/3AS se enlazaron con el padrón por `COD_MOD + ANEXO`; luego se agregaron por UBIGEO |

La matrícula es un conteo por servicio educativo, no una lista pública de alumnos con una llave individual reutilizable. La existencia de una llave escolar dentro de ESCALE no crea la misma llave en ENAHO. La tabla `outputs/tables/auditoria_cruce_matricula_2025.json` registra 54 038 servicios coincidentes de 54 038 (100 %); `matricula_escale_distrito_2025.csv` contiene el agregado distrital usado en el pipeline. La unión final con cada escolar ENAHO es por UBIGEO y representa contexto territorial; no asigna la escuela del estudiante. Una unión por proximidad sería una inferencia no respaldada por observación de asistencia.

## Verificación temporal y de descarga

El catálogo oficial coloca el archivo en la carpeta **Censo Educativo 2025** y registra actualización del **23/01/2026**. Esa es la fecha de actualización del catálogo; el año estadístico está identificado por la carpeta y diccionario de 2025. La ruta `/documents/inicio/Bases de datos/A. Censo Educativo/2025/01_Matricula_01.zip` permitió descargarlo tras el bloqueo de la ruta directa. El ZIP pasó su comprobación de integridad, contiene `Matricula_01.dbf` y su SHA-256 se registra en `data/raw/README.md`.

## Si preguntan por qué no se cruzó por escuela

La respuesta técnica es que faltan llaves comunes en el nivel persona. `COD_MOD` identifica un servicio educativo, no una persona; `CODGEO` identifica un distrito y no una institución concreta. Por ello la asociación a nivel distrital es reproducible, mientras que un enlace individual no sería verificable con estos insumos.

## Fuentes oficiales

- [Catálogo ESCALE: bases del Censo Educativo 2025](https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985)
- [Diccionario de datos del Censo Educativo 2025](https://escale.minedu.gob.pe/documents/10156/10632985/00_Diccionario%2Bde%2Bdatos_CE_2025.pdf)
- [Padrón de instituciones educativas ESCALE](https://escale.minedu.gob.pe/web/inicio/padron-de-iiee)
