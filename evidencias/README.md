# Paquete integral de evidencias para ChatGPT

## Propósito y fuente
Este paquete reúne evidencia verificable del proyecto `entregables/` ENAHO–ESCALE 2025 para redactar posteriormente un artículo según la guía del docente. No reinicia el análisis, no modifica fuentes y no exporta registros individuales. Se inspeccionó el dataset Parquet procesado, los scripts, notebook, tablas, modelo guardado, reportes y figuras. La guía PDF del docente está en la ruta Downloads compartida por el usuario; se considera el documento rector, aunque no se adjunta duplicada.

## Resumen numérico confirmado

| Evidencia | Resultado | Fuente dentro del paquete |
|---|---:|---|
| Base analítica | 17,992 filas × 109 columnas | `01_INGENIERIA_DATOS/01_resumen_dataset.json` |
| Prevalencia no ponderada del target | 25.64% | mismo JSON |
| Split agrupado | 14,407 train; 3,585 test; 3,292 y 824 conglomerados; cero compartidos | `01_INGENIERIA_DATOS/01_control_fuga_datos.json` |
| Clustering ganador guardado | KMeans k=2; Silhouette 0.307; DB 1.407; CH 750.36 | `02_CLUSTERING/02_resultados_clustering.json` |
| Clasificación RandomForest | test ROC-AUC 0.828; PR-AUC trapezoidal 0.571; F1 macro 0.712; precisión 0.483; recall 0.700 | `03_CLASIFICACION/03_resultados_supervisados.json` |
| Matriz de confusión no ponderada | TN 2,110; FP 591; FN 206; TP 678 | `03_CLASIFICACION/03_analisis_errores.json` |
| Cruce ESCALE-Padrón | 54,038 de 54,038 servicios enlazados por COD_MOD+ANEXO | `01_INGENIERIA_DATOS/auditoria_cruce_matricula_2025.json` |

PR-AUC en los resultados originales significa integral trapezoidal de precisión-recall. Average Precision calculada en la auditoría aparece aparte y no debe intercambiarse con esa métrica. Métricas de test de la tabla existente son las que el proyecto guardó; las curvas nuevas se derivaron del modelo y split existentes, sin entrenamiento.

## Hallazgos por fase

- **Datos:** la integración final cuenta con 17,992 observaciones y 109 variables. Códigos y llaves se leen como cadenas en el pipeline. La tabla de calidad reporta faltantes por campo. No se guardó en los artefactos un registro completo de conteos por cada filtro de construcción, así que esa secuencia no se declara auditada numéricamente aquí.
- **ESCALE:** matrícula se cruza con padrón institucional, pero ENAHO no tiene COD_MOD/CODLOCAL; la integración final es contexto por UBIGEO, no enlace de cada alumno a su centro. No hay evidencia de altitud, equipos o conectividad escolar.
- **Clustering:** KMeans y aglomerativo evaluados para k=2…6. KMeans k=2 domina por Silhouette entre resultados almacenados. Los nombres de perfil son descriptivos y exploratorios; cluster 0 concentra 89.8%, cluster 1 10.2%. La selección no valida grupos como categorías causales.
- **Supervisado:** regresión logística baseline y tres modelos avanzados comparados; RandomForest fue elegido por mayor ROC-AUC media en GroupKFold. HistGradientBoosting supera a RF en algunas métricas de test (ROC/PR trapezoidal), lo que debe mencionarse al discutir el criterio de selección. A umbral 0.5 hay precisión ~0.48, por lo que existen falsos positivos importantes. FACTOR07 y pesos de clase están documentados en código.
- **SHAP:** top global por magnitud agrupada: P308A, edad, uso de computadora, gasto per cápita, habilidades declaradas, ingreso per cápita, pobreza, ruralidad, grado, proporción de gestión pública. Contribuciones SHAP no son causales. La salida de TreeExplainer no fija explícitamente `model_output`; evita etiquetar sus unidades como probabilidad.

## Gráficas

| Archivo | Contenido/origen |
|---|---|
| `GRAFICAS_PNG/01_diagrama_integracion.png` | diagrama de uniones ENAHO y agregación UBIGEO, generado en esta auditoría, 300 DPI |
| `GRAFICAS_PNG/02_grafico_silueta.png` | figura silhouette del pipeline fuente, copiada sin cambios |
| `GRAFICAS_PNG/02_grafico_pca.png` | proyección PCA fuente |
| `GRAFICAS_PNG/02_comparacion_metricas.png` | tres métricas por algoritmo y k, desde CSV fuente, 300 DPI |
| `GRAFICAS_PNG/02_distribucion_clusters.png` | tamaños de perfiles del CSV fuente, 300 DPI |
| `GRAFICAS_PNG/03_matriz_confusion.png` | matriz de conteos guardada, sin ponderar, 300 DPI |
| `GRAFICAS_PNG/03_curvas_roc.png` | curva derivada de modelo/split guardados, ponderada |
| `GRAFICAS_PNG/03_curvas_precision_recall.png` | curva derivada; AP y PR trapezoidal diferenciadas |
| `GRAFICAS_PNG/03_curva_calibracion.png` | bins ponderados agregados; derivación de auditoría |
| `GRAFICAS_PNG/04_shap_beeswarm.png` | salida SHAP guardada, copiada |
| `GRAFICAS_PNG/04_shap_dependence_1.png`, `_2.png` | dependencias P1144/lengua y P1144/ratio distrito |
| `GRAFICAS_PNG/04_shap_waterfall_alto_riesgo.png`, `_bajo_riesgo.png` | extremos correctamente clasificados, fuente guardada; sin IDs en texto externo |
| `GRAFICAS_PNG/01_diagrama_integracion_origen.png` | diagrama fuente original |

Las figuras nuevas están también como SVG. Las figuras fuente se copiaron sin alterar resolución/estilo; solo las nuevas se exportaron a 300 DPI.

## Ubicación de JSON y CSV

| Evidencia | Ruta principal |
|---|---|
| Auditoría e inventario | `00_AUDITORIA/00_auditoria_proyecto.json` |
| Estado rúbrica | `00_AUDITORIA/00_estado_cumplimiento_rubrica.csv` |
| Resumen dataset | `01_INGENIERIA_DATOS/01_resumen_dataset.json` |
| Calidad / diccionario | `01_INGENIERIA_DATOS/01_calidad_datos.csv`, `01_diccionario_variables.csv` |
| Cruces / fuga | `01_INGENIERIA_DATOS/01_control_cruces.json`, `01_control_fuga_datos.json` |
| Clustering | `02_CLUSTERING/02_resultados_clustering.json`, `02_comparacion_algoritmos.csv`, `02_caracterizacion_clusters.csv` |
| Supervisado | `03_CLASIFICACION/03_resultados_supervisados.json`, `03_comparacion_modelos.csv`, `03_validacion_groupkfold.csv`, `03_hiperparametros.json`, `03_analisis_errores.json` |
| SHAP | `04_SHAP/04_resumen_shap.json`, `04_importancia_variables.csv`, `04_interpretaciones_shap.md` |
| Tablas artículo | `05_ARTICULO_CIENTIFICO/tabla_*.csv`, `05_leyendas_figuras.csv` |
| Manifiesto y dependencias | `06_REPRODUCIBILIDAD/06_manifest_archivos.json`, `06_dependencias.txt` |

## Cumplido con evidencia / pendiente

Cumplido verificablemente: integración procesada; auditoría de cruce matrícula-padrón; dos métodos de clustering y métricas; PCA; baseline y tres modelos; GroupKFold; split por CONGLOME sin intersección; matriz y métricas de clasificación; figuras SHAP existentes; diccionario y reportes. Parcial: auditoría de faltantes/exclusiones por etapa, calibración (derivada), análisis global del sentido SHAP, referencias bibliográficas institucionales completas y extensión/formato final IEEE/MDPI (el PDF fuente existente se incluye como referencia, pero no se afirma cumplimiento automático). Pendiente/ausente: altitud/equipos/conectividad escolar y enlace individual escolar. Consulte CSV de rúbrica; no interpretar estos vacíos como aprobados.

## Advertencias para el artículo

1. Decir “no uso de Internet en el mes previo”, no “incompetencia digital”.
2. Describir ESCALE como contexto de distrito; no atribuir indicadores distritales a la escuela individual del alumno.
3. Mencionar límites de variables escolares faltantes y agregación territorial.
4. Diferenciar PR-AUC trapezoidal de Average Precision; indicar ponderación.
5. Informar que RandomForest se eligió por CV, aunque HistGradientBoosting supera algunas métricas de test.
6. Calificar perfiles e interpretaciones SHAP como exploratorios/no causales.
7. No reportar identificadores de filas ni microdatos.
8. Referencias: completar datos bibliográficos con metadatos de los archivos institucionales; no añadir citas no verificadas.

## Archivos fuente clave

`../entregables/src/pipeline.py`, `../entregables/notebooks/01_evaluacion_brecha_digital.ipynb`, `../entregables/data/processed/dataset_enaho_escale_2025.parquet`, `../entregables/outputs/tables/`, `../entregables/outputs/models/metadatos_modelo.json`, `../entregables/outputs/figures/` y `../entregables/outputs/reports/`. El manifiesto incluye hashes SHA-256 y estado para cada evidencia generada.
