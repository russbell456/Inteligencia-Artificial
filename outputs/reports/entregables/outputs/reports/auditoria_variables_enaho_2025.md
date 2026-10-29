# Auditoría de variables ENAHO 2025

## Fuente

Se verificaron etiquetas y valores contra el diccionario oficial de los microdatos ENAHO 2025, descargado desde la página oficial de INEI y guardado en `data/raw/Diccionario_ENAHO_2025.pdf` (edición mayo de 2026). El enlace oficial está en [data/raw/README.md](../../data/raw/README.md).

## Variables utilizadas

| Variable | Etiqueta/codificación oficial | Uso en el proyecto |
|---|---|---|
| P306 | Matrícula en educación básica o superior durante el año: 1=sí, 2=no, 9=missing | Disponible en el módulo educativo; no se utiliza para definir la población analítica. |
| P307 | Asistencia actual: 1=sí, 2=no, 9=missing | Se filtra a 1; se excluye de los predictores. |
| P308A | Nivel matriculado: 1=inicial, 2=primaria, 3=secundaria, 4=superior no universitaria, 5=superior universitaria, 6=maestría/doctorado, 7=básica especial, 9=missing | Característica educativa del escolar. |
| P308B | Año de estudios matriculado; 9=missing | Característica educativa del escolar. |
| P314A | Uso de Internet durante el mes anterior: 1=sí, 2=no | Define el resultado `exclusion_digital`; se excluye de predictores. |
| P314B$1-$7 | Lugares declarados de uso de Internet | Descripción por opción; se excluye de predictores por ser condicional al uso. |
| P314D | Frecuencia declarada de uso de Internet | Descripción entre respuestas válidas; se excluye de predictores por su dependencia del resultado. |
| P316$1-$12 | Propósitos de uso de Internet, respuestas por ítem | Descripción (1=sí, 2=no); se excluye de predictores por condicionamiento/riesgo de fuga. |
| P316B | Uso de computadora/laptop/tableta en últimos tres meses | Predictor TIC exploratorio, distinto del objetivo P314A. |
| P316C1-C10 | Tareas digitales/informáticas declaradas, respuesta por ítem | Se resume como conteo de tareas afirmativas; predictor exploratorio, no evaluación certificada de competencia. |
| P1144 | Hogar tiene conexión a Internet fija/móvil: 0=no, 1=sí | Predictor de conectividad del hogar; conceptualmente distinto del uso individual P314A. |
| P1121 | Tipo de alumbrado del hogar: 0=pase/no electricidad, 1=electricidad | Predictor de servicio eléctrico; **no** se interpreta como acceso a Internet. |
| FACTOR07 | Factor de expansión anual ENAHO basado en proyecciones CPV-2007 | Ponderador aplicado cuando el estimador/modelo lo admite. |

El detalle por código, tamaño válido y faltantes (ponderado y sin ponderar) se genera en `outputs/tables/resumen_variables_tic_enaho.csv`. Los porcentajes de faltantes de la tabla analítica se conservan en `outputs/tables/diccionario_variables.csv`.

## Definición analítica confirmada

La población objetivo del modelo comprende personas de 6 a 17 años con P307=1 y P314A válido (1 o 2). La etiqueta es 1 cuando P314A=2 y 0 cuando P314A=1. Por ello, el proyecto evalúa no uso reciente de Internet entre asistentes; no mide competencias digitales ni ausencia de conectividad doméstica.

## Corrección de referencia

Se sustituyó una referencia metodológica anterior que apuntaba a un diccionario EPEN por el diccionario oficial ENAHO 2025. La ruta oficial correcta es `https://proyectos.inei.gob.pe/iinei/srienaho/Descarga/DocumentosMetodologicos/2025-55/20_Diccionario_2025.pdf`.
