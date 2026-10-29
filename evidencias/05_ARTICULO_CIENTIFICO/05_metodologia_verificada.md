# Metodología verificada

El pipeline fuente cruza ENAHO 100, 200, 300 y Sumaria mediante llaves de hogar/persona y conserva UBIGEO como texto. ESCALE matrícula 2025 se enlaza con el padrón institucional mediante COD_MOD+ANEXO; luego se resume por UBIGEO porque ENAHO disponible no contiene COD_MOD/CODLOCAL.

La población objetivo del clasificador se restringe a escolares asistentes de 6–17 años con respuesta válida P314A. El objetivo vale 1 si P314A=2 y 0 si P314A=1. Predictores y objetivo se separan en el código; el split y GroupKFold usan CONGLOME. Imputación, codificación y escalamiento pertenecen al pipeline scikit-learn, ajustado dentro de cada fold.

Clustering compara KMeans y Agglomerative sobre muestra de 4,000 con k entre 2 y 6; PCA apoya visualización. Clasificación compara regresión logística baseline con RandomForest, ExtraTrees e HistGradientBoosting. Se usan pesos de FACTOR07 cuando existen y pesos de clase según el código.
