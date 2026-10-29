# Resumen ejecutivo de auditoría

La carpeta fuente auditada es `entregables/`. Se localizaron pipeline Python, notebook Jupyter, base procesada, modelo RandomForest, resultados tabulares, figuras SHAP/clustering y un artículo PDF de 5 páginas. La salida analítica disponible contiene 17,992 filas y 109 columnas.

Los modelos y métricas persistidas respaldan clustering KMeans k=2 (Silhouette 0.307) y clasificación RandomForest con ROC-AUC test guardado 0.828 y F1 macro 0.712. Las curvas ROC, PR y calibración de este paquete se derivaron de la base y modelo guardado sin entrenamiento adicional.

El test está aislado por CONGLOME (3,292 grupos train; 824 test; cero compartidos). El cruce ESCALE se documenta a nivel UBIGEO; no vincula cada estudiante a una escuela. No hay evidencia de altitud, equipamiento informático ni conectividad escolar. La definición target mide no uso mensual reportado, no destrezas digitales.

La guía PDF de 7 páginas fue localizada y cotejada; contiene una rúbrica de 20 puntos. El artículo existente tiene 5 páginas. El archivo rector no se duplica en este paquete. La matriz registra criterios demostrados y pendientes, sin estimar una nota. Consultar `00_estado_cumplimiento_rubrica.csv` para estados y vacíos verificables.
