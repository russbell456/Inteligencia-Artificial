# Lectura de SHAP

Modelo explicado: RandomForest guardado. El código usa `shap.TreeExplainer(estimator)` y toma la salida de la clase positiva. No especifica `model_output`; por eso no se debe afirmar que las atribuciones estén en unidades de probabilidad.

La importancia global se calcula agrupando contribuciones codificadas por variable de origen y promediando el valor absoluto. Las diez variables aparecen en `04_importancia_variables.csv`; esto ordena magnitud, no dirección.

En los casos locales, SHAP positivo eleva la salida explicada respecto de la base del modelo y SHAP negativo la reduce. Los perfiles seleccionados son extremos correctamente clasificados del conjunto de prueba, anonimizados aquí como `caso_alto_01` y `caso_bajo_01`. No se exportan identificadores de fila.

Las dependencias almacenadas exploran P1144 (Internet en hogar), coloreada por P300A en la primera y ratio alumnos/docente distrital en la segunda. Son asociaciones del modelo; no efectos causales ni evidencia de competencia digital.
