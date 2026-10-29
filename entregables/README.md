# Entregables: Brecha digital y educación (ENAHO–ESCALE 2025)

## Contenido
- `data/raw/`: archivos ENAHO 100/200/300/Sumaria y ESCALE existentes usados, más diccionario ENAHO.
- `data/processed/`: dataset analítico CSV y Parquet y diccionario de variables.
- `notebooks/01_evaluacion_brecha_digital.ipynb`: notebook Jupyter reproducible.
- `src/`: pipeline de integración, clustering, clasificación, SHAP e informes.
- `outputs/tables/`: métricas, auditorías, matrices, resúmenes y perfiles. Incluye métricas ROC-AUC/PR-AUC/F1 macro por fold (`cv_groupkfold_modelos.csv`), auditoría de disyunción de conglomerados, distribución ponderada de clases y varianza explicada de PCA.
- `outputs/figures/`: cruce, clustering/PCA, silueta, evaluación y SHAP.
- `outputs/reports/`: artículo PDF de 5 páginas, versiones Markdown/HTML y limitaciones.
- `outputs/models/`: modelo ganador y metadatos.

## Abrir en Jupyter
Desde esta carpeta en PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m ipykernel install --prefix .venv --name enaho-escale-2025 --display-name "Python (ENAHO-ESCALE 2025)"
jupyter lab
```
Abre el notebook y elige el kernel `Python (ENAHO-ESCALE 2025)`. En la carpeta original ya se creó este entorno Jupyter; los comandos permiten recrearlo dentro del paquete. Para regenerar la tabla y los análisis: `python -m src.pipeline` (semilla 42). Para renderizar de nuevo el artículo PDF: `python -m src.create_article_pdf`.

## Definición y evaluación
La población son escolares asistentes de 6 a 17 años con respuesta válida sobre Internet. `exclusion_digital=1` significa que no reportó uso en el mes previo (P314A=2); no mide directamente competencias. Se comparan K-Means y Agglomerative; regresión logística como base frente a Random Forest, Extra Trees e HistGradientBoosting. La selección se realiza con GroupKFold y reporta ROC-AUC, PR-AUC y F1 macro por fold. La prueba se aísla por `CONGLOME`; existe una auditoría automática de cero grupos compartidos. `class_weight` atiende el desbalance y FACTOR07 pondera ajustes y métricas que admiten pesos. El test informa ROC/PR, F1, matrices de confusión, calibración y Brier score.

P314D se usa solo para clustering, con 0=no uso y 1–4=frecuencia observada; se excluye de clasificación. P316B y P316C son aproximaciones autodeclaradas de uso/tareas digitales, no pruebas certificadas de destreza.

## Alcance y límites que deben exponerse
Matrícula 2025 y Padron se enlazan por `COD_MOD + ANEXO`: 54.038/54.038 servicios se vincularon y se agregan por UBIGEO. `Padlocal.dbf` se usa para contar locales y auditar `CODLOCAL`; no tiene coordenadas numéricas ni altitud. ENAHO no trae `COD_MOD`/`CODLOCAL`, así que no se asigna cada escolar a su escuela. Los archivos incluidos tampoco contienen indicadores utilizables de altitud, equipos ni conectividad escolar. No se infirieron ni descargaron datos adicionales.

Consulta `outputs/tables/matriz_cumplimiento_guia.csv` para ver cada requisito, evidencia y limitación.


