# Brecha digital y educación en Perú (ENAHO–ESCALE 2025)

Proyecto reproducible. Los CSV ENAHO y DBF ESCALE fuente están en la raíz; no se copian ni modifican. El procesamiento crea `data/processed/` y los productos analíticos quedan en `outputs/`.

## Entorno Jupyter

En PowerShell, desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m ipykernel install --prefix .venv --name enaho-escale-2025 --display-name "Python (ENAHO-ESCALE 2025)"
jupyter lab
```

Abre `notebooks/01_evaluacion_brecha_digital.ipynb` y elige el kernel indicado. Por terminal: `python -m src.pipeline`.

## Diseño y límites

La unidad analítica son escolares de 6 a 17 años con asistencia reportada y respuesta válida sobre uso de Internet. `exclusion_digital` vale 1 si no reportó uso de Internet el mes anterior (P314A=2). Es un indicador de uso, no mide competencias digitales ni rezago de aprendizaje.

Los módulos personales se unen por AÑO, MES, CONGLOME, VIVIENDA, HOGAR, UBIGEO y CODPERSO; los hogares, por las mismas llaves sin CODPERSO. Los códigos se conservan como texto, y la lectura utiliza Latin-1 y reconoce celdas con espacios como faltantes. ESCALE se agrega por UBIGEO desde `Padron.dbf`. No hay identificador de matrícula para asignar cada persona a su escuela; las variables escolares son contexto distrital. `Padlocal.dbf` no contiene coordenadas.

### Verificación de ESCALE y llave de matrícula

El archivo local `Padron.dbf` contiene `COD_MOD` (servicio educativo), `CODLOCAL` (local), `CODGEO` (distrito), `TALUM`, `TDOC`, `TSEC` y coordenadas de la institución. El padrón es de servicios educativos y ofrece contexto escolar agregado; la coincidencia del estudiante ENAHO con una institución específica no se puede hacer porque sus módulos disponibles no contienen `COD_MOD`, `CODLOCAL` ni un identificador institucional equivalente.

Se descargó el ZIP oficial `data/raw/01_Matricula_01_2025.zip` desde la ruta alternativa oficial de ESCALE; el ZIP pasó la verificación de integridad y contiene `Matricula_01.dbf` (178 425 registros). El diccionario confirma que C201 de las cédulas 3AP/3AS registra matrícula por grado, sexo y turno. El pipeline conserva esas filas, suma primaria y secundaria por servicio usando `COD_MOD + ANEXO`, enlaza ambos códigos con el padrón y verifica la cobertura del cruce: los resultados quedan en `outputs/tables/auditoria_cruce_matricula_2025.json` y `matricula_escale_distrito_2025.csv`.

ENAHO continúa sin tener llave de escuela: la matrícula y el padrón enlazan entre sí, pero con ENAHO solo se pueden sumarizar por `CODGEO`/UBIGEO. Las nuevas variables son contexto distrital de matrícula en primaria y secundaria; no describen la matrícula ni la institución del escolar encuestado. No se asignan estudiantes a la escuela más cercana ni se infiere matrícula individual a partir del distrito.

Fuentes oficiales verificadas:

- [Catálogo ESCALE: bases de datos del Censo Educativo 2025](https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985)
- [Diccionario de datos del Censo Educativo 2025](https://escale.minedu.gob.pe/documents/10156/10632985/00_Diccionario%2Bde%2Bdatos_CE_2025.pdf)
- [Padrón de instituciones educativas ESCALE](https://escale.minedu.gob.pe/web/inicio/padron-de-iiee)

El test y los folds se separan por CONGLOME. Imputación, escalado y codificación están dentro de un Pipeline de scikit-learn. P307 y P314A no se usan como predictores. FACTOR07 pondera el ajuste cuando el estimador admite pesos. Los resultados son predictivos/descriptivos, no efectos causales.

Se incorporan las variables TIC P314B (lugar), P314D (frecuencia), P316 (propósitos), P316B (uso de computadora/tableta) y P316C1-C10 (tareas declaradas como aproximación de habilidades). Los resultados descriptivos y faltantes por código quedan en `outputs/tables/resumen_variables_tic_enaho.csv`. Lugar, frecuencia y propósito no entran al modelo por su condicionamiento al uso de Internet; P316B y el conteo P316C se exploran como predictores. La imputación incluye indicadores de faltante y se ajusta dentro de cada fold.

`Padlocal.dbf` sí está disponible en la carpeta, pero no contiene coordenadas numéricas. Se utilizan los agregados permitidos del Padrón y matrícula por distrito; no se infieren altitud, equipamiento ni conectividad escolar. La ausencia en ENAHO de COD_MOD/CODLOCAL impide atribuir una institución a cada escolar.

## Productos

- `src/pipeline.py`: cruces, variables derivadas, clustering, clasificación agrupada y SHAP.
- `notebooks/01_evaluacion_brecha_digital.ipynb`: bitácora ejecutable.
- `requirements.txt` y `requirements-lock.txt`: dependencias generales y versiones exactas del entorno generado.
- `data/processed/dataset_enaho_escale_2025.parquet`: tabla integrada (generada al ejecutar).
- `outputs/tables/`: diccionario, métricas, perfiles y resumen.
- `outputs/figures/`: PCA/clusters, curvas de evaluación y gráficos SHAP.
- `outputs/reports/`: informe técnico tras completar el análisis.
- `outputs/reports/articulo_brecha_digital_enaho_escale_2025.pdf`: artículo de cinco páginas con cruces, faltantes, prevención de fuga, clustering, métricas y SHAP.
- `outputs/reports/defensa_escal_llave_matricula.md`: evidencia, límites del enlace y guion breve para exponer la llave `COD_MOD`.
