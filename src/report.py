"""Genera el informe científico y diccionario desde los resultados calculados."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
R = ROOT / "outputs" / "reports"
R.mkdir(parents=True, exist_ok=True)


def generate_report() -> str:
    summary_path = T / "resumen_ejecucion.json"
    if not summary_path.exists():
        raise FileNotFoundError("Ejecute las celdas de integración, clustering y clasificación antes de generar el informe.")
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison_path = T / "comparacion_modelos.csv"
    clusters_path = T / "metricas_clustering.csv"
    comp = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    clusters = pd.read_csv(clusters_path) if clusters_path.exists() else pd.DataFrame()
    if len(comp):
        best = comp.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]
        model_table = comp.to_markdown(index=False, floatfmt=".3f")
        best_model = str(best.modelo)
        best_auc = f"{best.test_roc_auc:.3f}" if pd.notna(best.test_roc_auc) else "no estimable"
        best_pr = f"{best.test_pr_auc:.3f}" if pd.notna(best.test_pr_auc) else "no estimable"
        best_f1 = f"{best.test_f1_macro:.3f}" if pd.notna(best.test_f1_macro) else "no estimable"
    else:
        model_table, best_model, best_auc, best_pr, best_f1 = "Resultados no disponibles.", "no disponible", "no estimable", "no estimable", "no estimable"
    cluster_table = clusters.to_markdown(index=False, floatfmt=".3f") if len(clusters) else "Resultados no disponibles."
    profiles_path = T / "perfiles_clusters.csv"
    profiles = pd.read_csv(profiles_path) if profiles_path.exists() else pd.DataFrame()
    profile_table = profiles.to_markdown(index=False, floatfmt=".3f") if len(profiles) else "Perfiles no disponibles."
    if len(profiles) and "exclusion_digital" in profiles:
        high_profile = profiles.loc[profiles.exclusion_digital.idxmax()]
        low_profile = profiles.loc[profiles.exclusion_digital.idxmin()]
        profile_finding = (f"La mayor prevalencia observada de no uso se registró en el perfil **{high_profile.nombre_perfil}** "
                           f"(cluster {int(high_profile.iloc[0])}: {100*high_profile.exclusion_digital:.1f}%), frente a "
                           f"{100*low_profile.exclusion_digital:.1f}% en el perfil de menor prevalencia. Estas diferencias son descriptivas.")
    else:
        profile_finding = "No hay perfiles calculados para una comparación descriptiva."
    shap_path = T / "shap_importancia_top10.csv"
    shap_table = pd.read_csv(shap_path).to_markdown(index=False, floatfmt=".4f") if shap_path.exists() else "Importancia SHAP aún no disponible."
    tic_path = T / "resumen_variables_tic_enaho.csv"
    tic = pd.read_csv(tic_path) if tic_path.exists() else pd.DataFrame()
    skill_summary = ""
    if len(tic):
        skills = tic[tic.variable.str.startswith("P316C")]
        if len(skills):
            skill_summary = (f"En la muestra analítica, las preguntas P316C1–P316C10 suman {int(skills.n_valido.max()):,} respuestas válidas por ítem como máximo; "
                             "el porcentaje de sí por tarea se informa en `resumen_variables_tic_enaho.csv`. La suma individual de tareas se utiliza como predictor exploratorio; las no respuestas se conservan como faltantes e imputan dentro de cada fold con indicador de faltante.")
    dict_path=T/"diccionario_variables.csv"
    dictionary=pd.read_csv(dict_path) if dict_path.exists() else pd.DataFrame()
    focus=["P314A","P314D","tic_frecuencia_uso_cluster","rezago_escolar_anios","agua_potable","saneamiento_red_publica","tic_habilidades_digitales_n","tic_uso_computadora_ult3m","P1144","esc_ratio_alumno_docente","esc_matricula_primaria_2025","esc_n_locales_padlocal"]
    missing_table=(dictionary[dictionary.variable.isin(focus)][["variable","faltantes_pct"]]
                   .sort_values("variable").to_markdown(index=False,floatfmt=".1f")) if len(dictionary) else "Sin resumen de faltantes disponible."
    dependency_path = T / "shap_dependencias.json"
    dependencies = json.loads(dependency_path.read_text(encoding="utf-8")) if dependency_path.exists() else []
    local_path = T / "shap_casos_locales.json"
    local_cases = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else []
    local_summary = "No se generaron casos locales en esta ejecución."
    if local_cases:
        parts=[]
        for case in local_cases:
            top=", ".join(f"{x['variable']} ({x['shap']:+.3f})" for x in case["contribuciones_top"][:3])
            direction="elevó" if case["caso"]=="alto" else "redujo"
            parts.append(f"Caso {case['caso']} correctamente clasificado (real={case['real']}, predicho={case['predicho']}, probabilidad={case['probabilidad']:.3f}): entre las mayores contribuciones, {top}; los valores SHAP positivos {direction} la salida del modelo.")
        local_summary=" ".join(parts)
    cm_path = T / "matriz_confusion.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0).to_numpy()
        tn, fp, fn, tp = cm.ravel()
        error_summary = f"En el test agrupado, la matriz de conteos registra **{int(fp):,} falsos positivos** y **{int(fn):,} falsos negativos** (TN={int(tn):,}; TP={int(tp):,})."
    else:
        error_summary = "La matriz de confusión aún no está disponible."
    wcm_path = T / "matriz_confusion_ponderada.csv"
    if wcm_path.exists():
        wcm = pd.read_csv(wcm_path, index_col=0).to_numpy()
        wtn, wfp, wfn, wtp = wcm.ravel()
        error_summary += f" Con FACTOR07, los falsos positivos suman {wfp:,.1f} y los falsos negativos {wfn:,.1f} en escala ponderada."
    prevalence = 100 * s["target_prevalencia"]
    content = f'''# Brecha digital y educación en Perú: análisis exploratorio y predicción con ENAHO–ESCALE 2025

**Informe técnico tipo artículo científico · Corte de datos: ENAHO 2025 · Semilla: 42**

## Resumen

Este estudio construye un pipeline reproducible para integrar microdatos de hogares y personas de la Encuesta Nacional de Hogares (ENAHO) 2025 con información del padrón escolar ESCALE disponible en la carpeta de trabajo. La población analítica está conformada por personas de 6 a 17 años con asistencia escolar reportada y dato válido sobre uso de Internet. Se define un indicador observable de exclusión de uso digital como no haber utilizado Internet durante el mes anterior. La ejecución produjo **{s['n_filas_analiticas']:,} observaciones**, **{s['n_conglomerados']:,} conglomerados** y una prevalencia de resultado positivo de **{prevalence:.1f}%**. Se comparan perfiles no supervisados mediante K-Means y agrupamiento aglomerativo, y cuatro clasificadores mediante validación cruzada GroupKFold y prueba aislada por conglomerado. La mejor media de validación cruzada corresponde a **{best_model}**; en la prueba agrupada obtuvo ROC-AUC={best_auc}, PR-AUC={best_pr} y F1 macro={best_f1}. La interpretación debe circunscribirse al uso reciente de Internet: no equivale a una medición directa de competencias digitales ni identifica efectos causales.

**Palabras clave:** brecha digital; educación; ENAHO; ESCALE; aprendizaje automático; validación agrupada; explicabilidad.

## 1. Introducción

El acceso y uso de tecnologías de información pueden condicionar las oportunidades educativas, especialmente en hogares con menores recursos y en ámbitos rurales. Para orientar la priorización de intervenciones es necesario describir de manera conjunta las características de los escolares, sus hogares y el contexto de oferta educativa. La guía del proyecto propone integrar información ENAHO y ESCALE, descubrir tipologías con aprendizaje no supervisado y evaluar modelos predictivos con prevención explícita de fuga de información.

El objetivo de este análisis es construir y documentar un flujo verificable de preparación, caracterización, clasificación y explicación algorítmica con los archivos que estaban disponibles al ejecutar el proyecto. La pregunta predictiva concreta es: **entre escolares asistentes de 6 a 17 años con respuesta válida, ¿qué tan bien pueden ordenarse los casos según no haber usado Internet el mes previo usando características observadas de la persona, el hogar y el distrito?** Esta formulación evita presentar como variable observada una noción más amplia de competencia digital que los archivos no miden.

## 2. Datos y metodología

### 2.1 Fuentes y unidad de análisis

Se leyeron los módulos ENAHO 100 (vivienda/hogar), 200 (miembros), 300 (educación y TIC) y Sumaria 2025. El módulo personal se restringió a edades de 6 a 17 años y se enlazó por AÑO, MES, CONGLOME, VIVIENDA, HOGAR, UBIGEO y CODPERSO. Los archivos de hogar se enlazaron por las mismas llaves sin CODPERSO. Antes de unir, se preservaron como texto las llaves y los códigos geográficos para conservar ceros a la izquierda; la lectura usa codificación Latin-1 y convierte las celdas compuestas únicamente por espacios en valores ausentes. Los cruces tienen validaciones de cardinalidad para evitar duplicaciones silenciosas.

El padrón `Padron.dbf` se convirtió a registros tabulares y se resumió por UBIGEO. Se calcularon conteos y sumas de instituciones, alumnos, docentes y secciones; proporción administrativa pública; medias de coordenadas del padrón cuando son numéricas; y ratio alumnos/docente. La matrícula 2025 se enlazó al padrón por `COD_MOD + ANEXO` y se agregó por UBIGEO; la cobertura de ese cruce está en la auditoría. El `Padlocal.dbf` existente se incorporó para contar locales educativos por UBIGEO y auditar CODLOCAL contra `Padron.dbf`; no aporta coordenadas numéricas ni altitud. ENAHO tampoco tiene `COD_MOD`/`CODLOCAL`, así que ESCALE caracteriza el distrito y no la institución de cada estudiante. Los archivos existentes no permiten añadir altitud, equipos informáticos o conectividad escolar: no se imputan ni infieren.

El esquema de integración es: **Módulo 200 + Módulo 300** por llaves de persona → **Módulo 100 + Sumaria** por llaves de hogar → **indicadores ESCALE** por UBIGEO → tabla analítica filtrada por edad, asistencia y respuesta TIC.

![Diagrama de cruce de tablas](../figures/diagrama_cruce_tablas.png)

### 2.2 Variables y resultado

El resultado binario `exclusion_digital` vale 1 cuando P314A=2 (no usó Internet durante el mes anterior) y 0 cuando P314A=1. Se incluyen solamente casos con P307=1 (asistencia reportada) y P314A válido. La prevalencia observada en la muestra analítica es {prevalence:.1f}%. Se construyeron ingreso y gasto del hogar por miembro, ruralidad derivada del estrato, indicadores de agua potable, red pública de agua y saneamiento, electricidad doméstica, años de rezago escolar con edades normativas de primaria y secundaria, y características distritales de ESCALE. Los predictores categóricos y continuos se documentan en el diccionario de variables generado con el dataset.

Se incorporaron P314B (lugares de uso), P314D (frecuencia), P316 (propósitos), P316B (uso reciente de computadora/tablet) y P316C1–P316C10 (tareas informáticas que sirven como aproximación de habilidades digitales). Las distribuciones por código, faltantes y porcentajes ponderados se guardan en `outputs/tables/resumen_variables_tic_enaho.csv`. P314B y P316 se describen, y P314D se usa además como variable de perfilamiento no supervisado (0=no uso; 1–4=frecuencia observada). Las tres quedan excluidas de los predictores de clasificación por su condicionamiento al uso de Internet y riesgo de fuga. P316B y el conteo de tareas P316C se exploran como predictores, con imputación y marcadores de faltante dentro del pipeline. Este conteo representa tareas declaradas, no una certificación de competencia.

Los códigos se cotejaron con el Diccionario ENAHO 2025 disponible en el proyecto. P1121 representa electricidad doméstica y no se interpreta como Internet. El tratamiento de faltantes se evidencia en `outputs/tables/diccionario_variables.csv` (porcentaje por variable), `resumen_variables_tic_enaho.csv` (N válido y faltante por ítem) y el `SimpleImputer(add_indicator=True)` ajustado dentro de cada fold. En P316B y en el conteo P316C hay faltantes altos, consistentes con preguntas condicionadas y saltos del cuestionario; no se interpretan como respuesta "no". El pipeline conserva el faltante, añade el indicador correspondiente y lo imputa dentro de cada fold.

| Variable analítica | Faltantes (%) |
|---|---:|
{missing_table}

### 2.3 Prevención de fuga de datos y evaluación

Se fijó random_state=42. Un `GroupShuffleSplit` aisló el 20% de los datos por CONGLOME para prueba. Dentro del conjunto de entrenamiento se utilizó GroupKFold con hasta tres folds, de forma que observaciones del mismo conglomerado no aparezcan a la vez en ajuste y validación. La imputación mediana/moda, el escalado y la codificación one-hot están encapsulados en un `Pipeline` y se ajustan de nuevo dentro de cada fold. P307 y P314A se excluyeron de la matriz de predictores, pues se usan para determinar la población o el objetivo. Se reporta FACTOR07 en el ajuste de los modelos con soporte para pesos.

La comparación incluye regresión logística balanceada como línea base, Random Forest, Extra Trees e HistGradientBoosting. Para los tres modelos avanzados se compara una cuadrícula pequeña de hiperparámetros candidatos mediante GroupKFold ponderado; FACTOR07 pondera el ajuste y las métricas. Se calcula ROC-AUC para cada candidato y en prueba; adicionalmente PR-AUC, F1 macro, precisión y recall. Se generan matrices de confusión (conteos y ponderada), curva ROC, curva Precision–Recall y curva de calibración. Los valores de CV sirven para seleccionar candidatos, por lo que la estimación de desempeño final se prioriza en el test por conglomerado, mantenido aislado durante la selección.

### 2.4 Agrupamiento y explicabilidad

La muestra reproducible de hasta 4 000 observaciones se transforma con imputación, escalado y codificación. Las métricas se calculan en los primeros 20 componentes principales y la figura presenta dos componentes. Se probaron K-Means y AgglomerativeClustering para k=2…6, y la solución seleccionada también se muestra con su gráfico de silueta. Las variables incluyen ingreso y gasto per cápita, ratio alumnos/docente distrital, frecuencia TIC ordinal (P314D; 0 para no uso), años de rezago y servicios básicos. P314D entra solo al clustering; permanece excluida de predictores supervisados por ser una pregunta condicionada al uso. Los nombres exploratorios comparan ruralidad y no uso con el promedio analítico y se respaldan con las medias de la tabla, sin validación externa.

| Algoritmo | k | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---:|---:|---:|---:|
{cluster_table.replace(chr(10), chr(10))}

La composición se describe con conteo de casos y medias observadas, incluyendo ruralidad, Internet del hogar, habilidades declaradas y no uso del mes anterior:

{profile_table}

{profile_finding}

![Proyección PCA de los perfiles](../figures/clusters_pca.png)

![Gráfico de silueta de los perfiles](../figures/silhouette_clusters.png)

Para el modelo seleccionado se generan SHAP beeswarm, dependencia P1144 con la categoría de lengua materna y P1144 con ratio alumnos/docente distrital, además de dos waterfall de casos correctamente clasificados. SHAP explica contribuciones a la salida del modelo y no estima efectos causales.

## 3. Resultados

### 3.1 Desempeño de los modelos

La prevalencia positiva fue {prevalence:.1f}%, que debe contrastarse con la distribución de clases antes de interpretar accuracy o una única métrica. La clasificación se evaluó con separación por conglomerado y ponderación disponible. La Tabla 2 contiene los resultados calculados en esta ejecución. El modelo con mayor ROC-AUC medio GroupKFold fue **{best_model}**. Sus métricas en prueba agrupada fueron ROC-AUC={best_auc}, PR-AUC={best_pr} y F1 macro={best_f1}. La PR-AUC es especialmente informativa si hay desbalance y debe compararse con la prevalencia de la clase positiva.

| Modelo | AUC CV media | DE CV | AUC prueba | PR-AUC | F1 macro | Precisión | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
{model_table}

![Evaluación agrupada del clasificador](../figures/evaluacion_modelos.png)

La matriz de confusión y las curvas del modelo ganador se guardan junto al informe. Los falsos negativos representan escolares sin uso reciente de Internet que el modelo no priorizaría; los falsos positivos implican priorizar personas cuyo uso observado sí ocurrió. En una política de focalización, ambos errores tienen costos distintos y el umbral 0.5 no debe considerarse óptimo sin definir esos costos.

{error_summary}

### 3.2 Perfiles no supervisados

La Tabla 1 y los gráficos PCA y de silueta describen la solución seleccionada. Cada perfil recibe un nombre descriptivo derivado de su ruralidad y prevalencia relativa de no uso; la tabla permite contrastar también frecuencia TIC, ingreso/gasto, servicios básicos, rezago y contexto escolar. Son grupos exploratorios, no categorías oficiales ni segmentos de riesgo validados externamente.

### 3.3 Explicabilidad

Los artefactos SHAP muestran la contribución del modelo a su propia salida sobre los casos analizados. El beeswarm permite comparar importancia y dirección de asociación condicional; las dependencias ilustran relaciones aprendidas que pueden reflejar correlación entre atributos. Los waterfall son explicaciones locales para casos de la muestra de prueba y no deben generalizarse a todos los escolares. Si un artefacto SHAP no está disponible, el estado de generación se registra en `outputs/reports/shap_estado.txt`.

| Variable transformada | Media absoluta SHAP |
|---|---:|
{shap_table}

![SHAP global beeswarm](../figures/shap_beeswarm.png)

![Dependencia SHAP 1](../figures/shap_dependence_1.png)

![Dependencia SHAP 2](../figures/shap_dependence_2.png)

![Explicación local de alto riesgo](../figures/shap_waterfall_alto.png)

![Explicación local de bajo riesgo](../figures/shap_waterfall_bajo.png)

{skill_summary}

{local_summary}

## 4. Discusión

El flujo combina atributos individuales y domésticos con indicadores contextuales del padrón escolar, y aplica separación geográfica por conglomerado que reduce la posibilidad de que hogares cercanos o del mismo segmento muestral aparezcan a ambos lados de la evaluación. El pipeline de transformaciones evita que estadísticas de imputación, escalado o categorías del test se filtren al entrenamiento. Estas decisiones fortalecen la validez interna de la comparación predictiva.

Los resultados deben leerse con cautela por cuatro razones. Primero, el no uso de Internet en el mes previo no equivale a ausencia de conectividad, habilidades, calidad de conexión o aprendizaje. Segundo, el enlace ESCALE final es distrital; no prueba que el estudiante esté matriculado en una escuela con los valores promedio del distrito. Se incorporó la matrícula oficial Censo Educativo 2025: el cruce interno con el padrón por `COD_MOD + ANEXO` encontró 54 038 de 54 038 servicios. Los módulos ENAHO 2025 disponibles no tienen `COD_MOD` ni `CODLOCAL`, por lo que las variables nuevas describen el contexto distrital, no la escuela del escolar. Tercero, ENAHO es una encuesta con diseño complejo; los pesos se usan cuando el estimador lo admite, pero las métricas reportadas no constituyen necesariamente estimadores oficiales de prevalencia ni incorporan todas las etapas del diseño muestral. Cuarto, una asociación predictiva no identifica el efecto de ampliar infraestructura o entregar dispositivos.

La validez externa requiere repetir el análisis en otra muestra o edición. P316C permite describir tareas informáticas declaradas, pero no sustituye una evaluación de competencia digital ni de aprendizaje. Los archivos ESCALE existentes no contienen datos utilizables de altitud, equipos y conectividad escolar para este análisis; esos factores quedan como limitación explícita.

## 5. Conclusiones y recomendaciones

1. El proyecto deja una integración reproducible de los módulos ENAHO y el padrón ESCALE 2025 disponible, con llaves de texto y validación de cardinalidad; ESCALE se interpreta como contexto distrital porque ENAHO no proporciona la llave institucional requerida para el enlace individual.
2. La predicción del no uso de Internet se evalúa con separación y validación cruzada por CONGLOME; las transformaciones se ajustan dentro de cada fold.
3. El clustering aporta perfiles exploratorios y SHAP describe el comportamiento del clasificador seleccionado, sin convertir estos resultados en efectos causales.
4. La política educativa debería usar estos hallazgos como señal para diagnóstico territorial y validar la necesidad directamente con escuelas y hogares. No conviene asignar recursos individuales con una etiqueta predictiva sin evaluar errores, equidad y desempeño fuera de muestra.
5. Los archivos existentes no permiten atribuir escuela, altitud, equipamiento o conectividad escolar al estudiante; la vinculación institucional requiere una llave compatible en ENAHO y datos escolares pertinentes. P316C ofrece una aproximación de tareas digitales declaradas, no una evaluación directa de destreza.

## Referencias y reproducibilidad

- Instituto Nacional de Estadística e Informática (INEI). ENAHO 2025, módulos de microdatos y documentación metodológica. Portal oficial: https://www.inei.gob.pe/bases-de-datos/.
- Instituto Nacional de Estadística e Informática (INEI). *Diccionario de Datos ENAHO 2025* (archivo oficial descargado en `data/raw/Diccionario_ENAHO_2025.pdf`, edición mayo 2026): https://proyectos.inei.gob.pe/iinei/srienaho/Descarga/DocumentosMetodologicos/2025-55/20_Diccionario_2025.pdf.
- Ministerio de Educación del Perú. ESCALE, padrón de instituciones educativas y Censo Educativo. Portal oficial: https://escale.minedu.gob.pe/.
- Ministerio de Educación del Perú. Catálogo oficial de bases del Censo Educativo 2025 (incluye `00_Padron.zip` y `01_Matricula_01.zip`, actualizado en el catálogo el 23/01/2026): https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985.
- Ministerio de Educación del Perú. *Diccionario de datos del Censo Educativo 2025* (`COD_MOD`: identificador del servicio educativo): https://escale.minedu.gob.pe/documents/10156/10632985/00_Diccionario%2Bde%2Bdatos_CE_2025.pdf.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

**Reproducibilidad:** código en `src/pipeline.py`; notebook en `notebooks/01_evaluacion_brecha_digital.ipynb`; dependencias en `requirements.txt`; semilla 42. El resumen de datos y la huella de evaluación están en `outputs/tables/resumen_ejecucion.json` y `outputs/models/metadatos_modelo.json`.
'''
    out = R / "informe_tecnico_brecha_digital_2025.md"
    out.write_text(content, encoding="utf-8")
    # HTML autocontenido para lectura rápida; mantiene el Markdown como fuente editable.
    try:
        import markdown
        html = "<!doctype html><html lang='es'><meta charset='utf-8'><title>Informe ENAHO–ESCALE 2025</title><style>body{font:16px/1.55 Georgia,serif;max-width:960px;margin:3rem auto;padding:0 2rem;color:#18212b}h1,h2,h3{font-family:Arial,sans-serif;color:#17324d}table{border-collapse:collapse;width:100%;font:14px Arial}td,th{border:1px solid #ccd5df;padding:.45rem}th{background:#eef3f8}code{background:#f1f3f5;padding:.1rem .25rem}</style><body>" + markdown.markdown(content, extensions=["tables"]) + "</body></html>"
        (R / "informe_tecnico_brecha_digital_2025.html").write_text(html, encoding="utf-8")
    except ImportError:
        pass
    return str(out)


