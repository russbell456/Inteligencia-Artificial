# Resultados interpretados con trazabilidad

La base analítica tiene 17,992 filas y 109 columnas; la prevalencia no ponderada de `exclusion_digital` es 25.64%. La definición operacional es no reportar uso de Internet en el mes anterior entre escolares asistentes de 6–17 años con respuesta válida; no mide destrezas digitales directamente. Fuente: `01_INGENIERIA_DATOS/01_resumen_dataset.json`.

## Clustering

Se compararon Agglomerative, KMeans, k=2–6, con muestra de 4,000 y semilla 42. La fila de mayor Silhouette es KMeans k=2 (Silhouette 0.307; Davies–Bouldin 1.407; Calinski–Harabasz 750.4). Los nombres de perfiles son exploratorios; las etiquetas no prueban tipologías sociales estables. Fuente: `02_CLUSTERING/02_resultados_clustering.json`.

## Clasificación

RandomForest fue seleccionado por media ROC-AUC de GroupKFold (0.819; DE 0.011). En test agrupado registró ROC-AUC 0.828, PR-AUC trapezoidal 0.571, F1 macro 0.712, precisión 0.483, recall 0.700 y Brier 0.154 en tabla existente. La curva derivada del modelo guardado confirma AUC ponderada 0.828; Average Precision ponderada 0.572 se reporta por separado. La matriz no ponderada tuvo TN=2,110, FP=591, FN=206, TP=678. La precisión limita la confianza de alertas positivas al umbral 0.5. Fuente: `03_CLASIFICACION/03_resultados_supervisados.json` y `03_analisis_errores.json`.

## Explicabilidad

El ranking SHAP global inicia con P308A, edad, uso de computadora, gasto per cápita e índice de habilidades autodeclaradas. Son atribuciones del modelo, no efectos causales. La escala explicada no queda fijada explícitamente en el código y la dirección global por valor de variable no está disponible en la tabla agregada. Fuente: `04_SHAP/04_resumen_shap.json`.
