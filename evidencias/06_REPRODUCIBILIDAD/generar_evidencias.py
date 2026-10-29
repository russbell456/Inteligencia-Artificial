from pathlib import Path
import json, shutil, hashlib, csv, sys, platform, importlib.metadata as md
from datetime import datetime, timezone
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
import joblib

BASE=Path(__file__).resolve().parents[2]
SRC=BASE/'entregables'; OUT=Path(__file__).resolve().parents[1]
T=SRC/'outputs'/'tables'; F=SRC/'outputs'/'figures'
for d in ['00_AUDITORIA','01_INGENIERIA_DATOS','02_CLUSTERING','03_CLASIFICACION','04_SHAP','05_ARTICULO_CIENTIFICO','06_REPRODUCIBILIDAD','GRAFICAS_PNG','TABLAS_CSV','JSON_COMPLETOS']:
    (OUT/d).mkdir(parents=True,exist_ok=True)
now=datetime.now(timezone.utc).isoformat()
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def cp(src,dst):
    Path(dst).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def savefig(fig,name):
    p=OUT/'GRAFICAS_PNG'/name; fig.savefig(p,dpi=300,bbox_inches='tight'); fig.savefig(p.with_suffix('.svg'),bbox_inches='tight'); plt.close(fig); return p
# Copy existing aggregate evidence only; exclude student-level sample, model binary, raw/processed microdata.
copy_map={
 '00_AUDITORIA': ['matriz_cumplimiento_guia.csv'],
 '01_INGENIERIA_DATOS':['diccionario_variables.csv','resumen_variables_tic_enaho.csv','schema_dataset.csv','auditoria_cruce_matricula_2025.json','auditoria_cruce_padlocal.json','resumen_ejecucion.json'],
 '02_CLUSTERING':['metricas_clustering.csv','perfiles_clusters.csv','pca_varianza.csv','configuracion_clustering.json'],
 '03_CLASIFICACION':['comparacion_modelos.csv','cv_groupkfold_modelos.csv','distribucion_clases.csv','matriz_confusion.csv','matriz_confusion_ponderada.csv','reporte_clasificacion.txt','auditoria_split_conglomerado.json','metadatos_modelo.json'],
 '04_SHAP':['shap_importancia_top10.csv','shap_casos_locales.json','shap_dependencias.json'],
}
for folder,files in copy_map.items():
    for name in files:
        src=T/name if (T/name).exists() else SRC/'outputs'/'models'/name
        if src.exists(): cp(src,OUT/folder/name); cp(src,OUT/'TABLAS_CSV'/name) if src.suffix=='.csv' else None; cp(src,OUT/'JSON_COMPLETOS'/name) if src.suffix=='.json' else None
# Original figures preserved as provided (180 dpi according to source generation code).
figure_map={'clusters_pca.png':'02_grafico_pca.png','silhouette_clusters.png':'02_grafico_silueta.png','shap_beeswarm.png':'04_shap_beeswarm.png','shap_dependence_1.png':'04_shap_dependence_1.png','shap_dependence_2.png':'04_shap_dependence_2.png','shap_waterfall_alto.png':'04_shap_waterfall_alto_riesgo.png','shap_waterfall_bajo.png':'04_shap_waterfall_bajo_riesgo.png','diagrama_cruce_tablas.png':'01_diagrama_integracion_origen.png'}
for a,b in figure_map.items():
    if (F/a).exists(): cp(F/a,OUT/'GRAFICAS_PNG'/b)
# Dataset aggregate audit. No row-level records are exported.
df=pd.read_parquet(SRC/'data'/'processed'/'dataset_enaho_escale_2025.parquet')
summary=readj(T/'resumen_ejecucion.json')
key=[c for c in ['AÑO','MES','CONGLOME','VIVIENDA','HOGAR','UBIGEO','CODPERSO'] if c in df.columns]
nulls=df.isna().sum().sort_values(ascending=False)
summary_ds={'estado':'verificado desde Parquet procesado','archivo_origen':'entregables/data/processed/dataset_enaho_escale_2025.parquet','filas':int(df.shape[0]),'columnas':int(df.shape[1]),'duplicados_exactos':int(df.duplicated().sum()),'duplicados_llave_persona':int(df.duplicated(key).sum()) if key else None,'llave_persona_usada':key,'nulos_totales':int(df.isna().sum().sum()),'columnas_con_nulos':int((nulls>0).sum()),'top_nulos':{str(k):int(v) for k,v in nulls.head(20).items()},'target':'exclusion_digital: población escolar 6–17 años asistente (P307=1) con respuesta válida P314A; 1=P314A 2 (sin uso reciente), 0=P314A 1','prevalencia_target':float(df['exclusion_digital'].mean()),'conglomerados':int(df['grupo_conglomerado'].astype(str).nunique()),'fuentes_integradas':['ENAHO módulos 100, 200, 300, Sumaria','ESCALE matrícula 2025, Padrón, Padlocal, agregación distrital UBIGEO'],'limitacion_registros_fuente':'No se recalcularon aquí los conteos completos por etapa desde los CSV fuente; se preservan filtros descritos por pipeline y se registran los conteos no disponibles como pendiente.'}
writej(OUT/'01_INGENIERIA_DATOS'/'01_resumen_dataset.json',summary_ds)
quality=pd.DataFrame({'variable':df.columns,'tipo':[str(df[c].dtype) for c in df.columns],'nulos':[int(df[c].isna().sum()) for c in df.columns],'porcentaje_nulos':[float(df[c].isna().mean()*100) for c in df.columns],'n_unicos':[int(df[c].nunique(dropna=True)) for c in df.columns]})
quality.to_csv(OUT/'01_INGENIERIA_DATOS'/'01_calidad_datos.csv',index=False,encoding='utf-8-sig')
cp(SRC/'data'/'processed'/'diccionario_variables.csv',OUT/'01_INGENIERIA_DATOS'/'01_diccionario_variables.csv')
# Cross controls based on source code and saved audit records
cross={'ENAHO':{'llaves_persona':['AÑO','MES','CONGLOME','VIVIENDA','HOGAR','CODPERSO'],'llave_hogar':['AÑO','MES','CONGLOME','VIVIENDA','HOGAR'],'llave_sumaria':['AÑO','MES','CONGLOME','VIVIENDA','HOGAR'],'preservacion_ids':'pipeline lee códigos como texto; CONGLOME se conserva como string'},'ESCALE':readj(T/'auditoria_cruce_matricula_2025.json'),'Padlocal':readj(T/'auditoria_cruce_padlocal.json'),'vinculo_escolar_individual':False,'justificacion':'No existe COD_MOD/CODLOCAL en ENAHO 2025 disponible; matrícula y contexto se agregan por UBIGEO.','n_registros_analiticos':int(len(df))}
writej(OUT/'01_INGENIERIA_DATOS'/'01_control_cruces.json',cross)
split=readj(T/'auditoria_split_conglomerado.json')
leak={'split':split,'separacion_conglomerados_confirmada':bool(split.get('separacion_confirmada')),'preprocesamiento':'Pipeline scikit-learn aplica imputación, one-hot y escalado dentro del estimador/folds según código src/pipeline.py.','target_excluido_de_predictores':'el código define predictores sin P307 ni P314A; exclusión de campos outcome confirmada por función feature_columns.','variables_postresultado':'P314B/P316 descriptivas; P314D solo clustering; no usadas en clasificación según código y resumen de ejecución.','limitacion_verificacion':'Se verifican código y registro de split guardado; no se rehace el entrenamiento ni se exportan predicciones individuales.'}
writej(OUT/'01_INGENIERIA_DATOS'/'01_control_fuga_datos.json',leak)
# Derive integration diagram
fig,ax=plt.subplots(figsize=(13,6)); ax.axis('off')
boxes=[(.04,.65,.22,.23,'ENAHO 100\nHogar: llave compuesta'),(.30,.65,.22,.23,'ENAHO 200 + 300\nPersona: llave compuesta'),(.56,.65,.19,.23,'Sumaria\nHogar: llave compuesta'),(.79,.65,.17,.23,'ESCALE 2025\nCOD_MOD + ANEXO'),(.39,.18,.28,.25,'Dataset analítico\nENAHO + UBIGEO agregado\n17 992 × 109')]
for x,y,w,h,s in boxes: ax.add_patch(plt.Rectangle((x,y),w,h,fc='#eaf2f8',ec='#1f4e79',lw=1.8)); ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=10)
for x1,y1,x2,y2,label in [(.26,.76,.39,.43,'llave hogar'),(.52,.76,.48,.43,'llave hogar'),(.67,.65,.60,.43,'UBIGEO\n(distrito)'),(.87,.65,.64,.43,'agregación UBIGEO')]: ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.6,color='#555')); ax.text((x1+x2)/2,(y1+y2)/2,label,fontsize=8,ha='center')
ax.text(.5,.03,'ENAHO no contiene llave de institución; no se asigna una escuela a cada estudiante.',ha='center',fontsize=10,color='#9b1c1c'); ax.set_title('Integración documentada de ENAHO y ESCALE 2025'); savefig(fig,'01_diagrama_integracion.png')
# Clustering tables and figures
cl=pd.read_csv(T/'metricas_clustering.csv'); profiles=pd.read_csv(T/'perfiles_clusters.csv'); config=readj(T/'configuracion_clustering.json')
metric_obj={'fuente':'outputs/tables/metricas_clustering.csv','algoritmos':sorted(cl.algoritmo.unique().tolist()),'k_evaluados':sorted(cl.k.unique().tolist()),'seleccion_observada':'KMeans, k=2 (máximo Silhouette entre filas guardadas; DB y CH también se informan)','ganador':cl.sort_values('silhouette',ascending=False).iloc[0].to_dict(),'n_muestra':config.get('muestra_n'),'semilla':config.get('semilla'),'variables':config.get('variables_numericas',[]),'perfiles':profiles.to_dict(orient='records'),'limitacion':'Muestra de 4,000 y perfiles exploratorios; nombres no inferenciales. Algunas variables de ESCALE son distritales y faltan altitud/equipos/conectividad escolar.'}
writej(OUT/'02_CLUSTERING'/'02_resultados_clustering.json',metric_obj)
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for ax,col,title,ylabel in zip(axs,['silhouette','davies_bouldin','calinski_harabasz'],['Silhouette (más alto es mejor)','Davies–Bouldin (más bajo es mejor)','Calinski–Harabasz (más alto es mejor)'],['Coeficiente','Índice','Índice']):
 for alg,g in cl.groupby('algoritmo'):
  ax.plot(g.k,g[col],marker='o',label=alg)
 ax.set_title(title); ax.set_xlabel('Número de clusters k'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
fig.suptitle('Comparación de métricas de clustering'); savefig(fig,'02_comparacion_metricas.png')
profiles.to_csv(OUT/'02_CLUSTERING'/'02_caracterizacion_clusters.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(8,5)); ax.bar(profiles.cluster.astype(str),profiles.n,color=['#3572A5','#E07A3F']);
for i,row in profiles.iterrows(): ax.text(i,row.n,f"n={int(row.n):,}\n{row.n/profiles.n.sum():.1%}",ha='center',va='bottom')
ax.set_title('Tamaño de los perfiles KMeans (muestra de clustering)'); ax.set_xlabel('Cluster'); ax.set_ylabel('Número de observaciones'); ax.grid(axis='y',alpha=.2); savefig(fig,'02_distribucion_clusters.png')
cp(OUT/'02_CLUSTERING'/'02_caracterizacion_clusters.csv',OUT/'TABLAS_CSV'/'02_caracterizacion_clusters.csv')
# Supervised: reuse saved model and exact split only to derive aggregate test curves; no training.
sys.path.insert(0,str(SRC))
# module import from package path
if str(SRC.parent) not in sys.path: sys.path.insert(0,str(SRC.parent))
from entregables.src.pipeline import feature_columns as fc
num,cat=fc(df,include_digital_skills=True)
X=df[num+cat]; y=df['exclusion_digital'].astype(int); groups=df['grupo_conglomerado'].astype(str)
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X,y,groups))
model=joblib.load(SRC/'outputs'/'models'/'modelo_ganador.joblib')
prob=model.predict_proba(X.iloc[te])[:,1]; yt=y.iloc[te].to_numpy(); pred=(prob>=.5).astype(int)
w=pd.to_numeric(df.iloc[te]['factor_expansion'],errors='coerce'); w=w.fillna(w.median() if w.notna().any() else 1).clip(lower=0).to_numpy()
roc_x,roc_y,_=roc_curve(yt,prob,sample_weight=w); roc_auc=auc(roc_x,roc_y)
pr_p,pr_r,_=precision_recall_curve(yt,prob,sample_weight=w); pr_auc=auc(pr_r,pr_p); ap=average_precision_score(yt,prob,sample_weight=w)
bins=np.linspace(0,1,11); bid=np.minimum(np.digitize(prob,bins[1:-1]),9); mean=np.array([np.average(prob[bid==b],weights=w[bid==b]) for b in range(10) if np.any(bid==b)]); frac=np.array([np.average(yt[bid==b],weights=w[bid==b]) for b in range(10) if np.any(bid==b)])
# Save curve coordinates only (aggregated; not individual predictions)
pd.DataFrame({'fpr':roc_x,'tpr':roc_y}).to_csv(OUT/'03_CLASIFICACION'/'03_curva_roc_puntos.csv',index=False)
pd.DataFrame({'recall':pr_r,'precision':pr_p}).to_csv(OUT/'03_CLASIFICACION'/'03_curva_pr_puntos.csv',index=False)
fig,ax=plt.subplots(figsize=(6,5)); ax.plot(roc_x,roc_y,label=f'RandomForest (AUC ponderada={roc_auc:.3f})'); ax.plot([0,1],[0,1],'--',color='gray',label='Azar'); ax.set(xlabel='Tasa de falsos positivos',ylabel='Tasa de verdaderos positivos',title='Curva ROC en test por conglomerado'); ax.legend(); ax.grid(alpha=.2); savefig(fig,'03_curvas_roc.png')
fig,ax=plt.subplots(figsize=(6,5)); ax.plot(pr_r,pr_p,label=f'PR AUC trapezoidal={pr_auc:.3f}; AP={ap:.3f}'); ax.axhline(np.average(yt,weights=w),ls='--',color='gray',label='Prevalencia ponderada'); ax.set(xlabel='Recall',ylabel='Precisión',title='Curva precisión–recall en test'); ax.legend(); ax.grid(alpha=.2); savefig(fig,'03_curvas_precision_recall.png')
fig,ax=plt.subplots(figsize=(6,5)); ax.plot(mean,frac,marker='o',label='RandomForest'); ax.plot([0,1],[0,1],'--',color='gray',label='Calibración perfecta'); ax.set(xlabel='Probabilidad media predicha',ylabel='Fracción positiva observada',title='Calibración en test (10 bins)'); ax.legend(); ax.grid(alpha=.2); savefig(fig,'03_curva_calibracion.png')
cm=confusion_matrix(yt,pred,labels=[0,1]); fig,ax=plt.subplots(figsize=(5.5,4.8)); im=ax.imshow(cm,cmap='Blues');
for (i,j),v in np.ndenumerate(cm): ax.text(j,i,f'{v:,}',ha='center',va='center',color='black')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Predicho 0','Predicho 1'],yticklabels=['Real 0','Real 1'],title='Matriz de confusión (recuento sin ponderar)'); fig.colorbar(im,ax=ax,label='Observaciones'); savefig(fig,'03_matriz_confusion.png')
models=pd.read_csv(T/'comparacion_modelos.csv'); cvs=pd.read_csv(T/'cv_groupkfold_modelos.csv')
writej(OUT/'03_CLASIFICACION'/'03_resultados_supervisados.json',{'target':summary_ds['target'],'poblacion':{'n_total_analitico':int(len(df)),'n_test':int(len(te)),'n_train':int(len(tr)),'prevalencia_test_no_ponderada':float(yt.mean()),'prevalencia_test_ponderada':float(np.average(yt,weights=w))},'modelos':models.to_dict(orient='records'),'criterio_seleccion':'RandomForest por mayor media ROC-AUC ponderada en GroupKFold (según tabla del proyecto); HistGradientBoosting presenta métricas test mayores en ROC-AUC/PR trapezoidal, por lo que la selección no maximiza todos los criterios de test.','test_curves_recalculadas_desde_modelo_y_split_guardados':{'ROC_AUC_ponderada':float(roc_auc),'PR_AUC_trapezoidal_ponderada':float(pr_auc),'average_precision_ponderada':float(ap),'Brier_ponderado':float(brier_score_loss(yt,prob,sample_weight=w))},'definicion_PR':'PR-AUC informado por pipeline = integral trapezoidal auc(recall, precision); Average Precision se entrega aparte y no es intercambiable.','semilla':42,'split':split,'sin_reentrenamiento':True})
# CV and model parameter evidence
cp(T/'cv_groupkfold_modelos.csv',OUT/'03_CLASIFICACION'/'03_validacion_groupkfold.csv')
writej(OUT/'03_CLASIFICACION'/'03_hiperparametros.json',{'modelo_ganador':readj(SRC/'outputs'/'models'/'metadatos_modelo.json'),'modelos_comparados':models[['modelo','hiperparametros_seleccionados']].to_dict(orient='records'),'criterio':'Los valores se transcriben de los artefactos guardados; no se ajustaron parámetros en esta auditoría.'})
# Errors aggregate from actual saved confusion counts
fp=int(cm[0,1]); fn=int(cm[1,0]); tn=int(cm[0,0]); tp=int(cm[1,1])
writej(OUT/'03_CLASIFICACION'/'03_analisis_errores.json',{'umbral':0.5,'matriz_confusion_recuento':{'TN':tn,'FP':fp,'FN':fn,'TP':tp},'precision':float(tp/(tp+fp)),'recall':float(tp/(tp+fn)),'lectura':'El umbral produce más falsos positivos que falsos negativos en conteo no ponderado. La precisión (~0.48) y recall (~0.70) reflejan un compromiso; no existe umbral de desempeño de rúbrica para declarar aprobado/reprobado. No se exportan registros individuales.'})
# SHAP evidence, copied from outputs; private row indices excluded from evidence JSON.
shapj=readj(T/'shap_casos_locales.json'); anon=[]
for x in shapj:
 anon.append({k:v for k,v in x.items() if k!='indice_fila'} | {'caso':'caso_alto_01' if x.get('caso')=='alto' else 'caso_bajo_01'})
writej(OUT/'04_SHAP'/'04_resumen_shap.json',{'modelo':'RandomForest ganador guardado','explainer':'TreeExplainer por código del pipeline; valores de clase positiva vals[...,1]. model_output no se fija explícitamente.','fondo':'El código construye muestra bg de 100 filas; TreeExplainer se inicializa solo con estimador, por lo que esa muestra no se pasa como background al explainer.','muestra_explicada':'hasta 250 filas aleatorias del test más extremos correctos alto/bajo; casos almacenados son anonimizados.','importancia_global':pd.read_csv(T/'shap_importancia_top10.csv').to_dict(orient='records'),'casos_locales':anon,'dependencias':readj(T/'shap_dependencias.json'),'interpretacion':'SHAP describe atribuciones del modelo. El signo indica aumento/disminución de la salida explicada del TreeExplainer; no se interpreta como causalidad. Globalmente solo es verificable el orden por magnitud absoluta, no el sentido por valores de atributo.'})
imp=pd.read_csv(T/'shap_importancia_top10.csv'); imp.to_csv(OUT/'04_SHAP'/'04_importancia_variables.csv',index=False,encoding='utf-8-sig')
mdtxt='''# Lectura de SHAP\n\nModelo explicado: RandomForest guardado. El código usa `shap.TreeExplainer(estimator)` y toma la salida de la clase positiva. No especifica `model_output`; por eso no se debe afirmar que las atribuciones estén en unidades de probabilidad.\n\nLa importancia global se calcula agrupando contribuciones codificadas por variable de origen y promediando el valor absoluto. Las diez variables aparecen en `04_importancia_variables.csv`; esto ordena magnitud, no dirección.\n\nEn los casos locales, SHAP positivo eleva la salida explicada respecto de la base del modelo y SHAP negativo la reduce. Los perfiles seleccionados son extremos correctamente clasificados del conjunto de prueba, anonimizados aquí como `caso_alto_01` y `caso_bajo_01`. No se exportan identificadores de fila.\n\nLas dependencias almacenadas exploran P1144 (Internet en hogar), coloreada por P300A en la primera y ratio alumnos/docente distrital en la segunda. Son asociaciones del modelo; no efectos causales ni evidencia de competencia digital.\n'''
(OUT/'04_SHAP'/'04_interpretaciones_shap.md').write_text(mdtxt,encoding='utf-8')
# Article prep
narr=f'''# Resultados interpretados con trazabilidad\n\nLa base analítica tiene {len(df):,} filas y {df.shape[1]} columnas; la prevalencia no ponderada de `exclusion_digital` es {df.exclusion_digital.mean():.2%}. La definición operacional es no reportar uso de Internet en el mes anterior entre escolares asistentes de 6–17 años con respuesta válida; no mide destrezas digitales directamente. Fuente: `01_INGENIERIA_DATOS/01_resumen_dataset.json`.\n\n## Clustering\n\nSe compararon {', '.join(sorted(cl.algoritmo.unique()))}, k={min(cl.k)}–{max(cl.k)}, con muestra de {config.get('muestra_n'):,} y semilla {config.get('semilla')}. La fila de mayor Silhouette es KMeans k=2 (Silhouette {metric_obj['ganador']['silhouette']:.3f}; Davies–Bouldin {metric_obj['ganador']['davies_bouldin']:.3f}; Calinski–Harabasz {metric_obj['ganador']['calinski_harabasz']:.1f}). Los nombres de perfiles son exploratorios; las etiquetas no prueban tipologías sociales estables. Fuente: `02_CLUSTERING/02_resultados_clustering.json`.\n\n## Clasificación\n\nRandomForest fue seleccionado por media ROC-AUC de GroupKFold (0.819; DE 0.011). En test agrupado registró ROC-AUC 0.828, PR-AUC trapezoidal 0.571, F1 macro 0.712, precisión 0.483, recall 0.700 y Brier 0.154 en tabla existente. La curva derivada del modelo guardado confirma AUC ponderada {roc_auc:.3f}; Average Precision ponderada {ap:.3f} se reporta por separado. La matriz no ponderada tuvo TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}. La precisión limita la confianza de alertas positivas al umbral 0.5. Fuente: `03_CLASIFICACION/03_resultados_supervisados.json` y `03_analisis_errores.json`.\n\n## Explicabilidad\n\nEl ranking SHAP global inicia con P308A, edad, uso de computadora, gasto per cápita e índice de habilidades autodeclaradas. Son atribuciones del modelo, no efectos causales. La escala explicada no queda fijada explícitamente en el código y la dirección global por valor de variable no está disponible en la tabla agregada. Fuente: `04_SHAP/04_resumen_shap.json`.\n'''
(OUT/'05_ARTICULO_CIENTIFICO'/'05_resultados_interpretados.md').write_text(narr,encoding='utf-8')
(OUT/'05_ARTICULO_CIENTIFICO'/'05_metodologia_verificada.md').write_text('''# Metodología verificada\n\nEl pipeline fuente cruza ENAHO 100, 200, 300 y Sumaria mediante llaves de hogar/persona y conserva UBIGEO como texto. ESCALE matrícula 2025 se enlaza con el padrón institucional mediante COD_MOD+ANEXO; luego se resume por UBIGEO porque ENAHO disponible no contiene COD_MOD/CODLOCAL.\n\nLa población objetivo del clasificador se restringe a escolares asistentes de 6–17 años con respuesta válida P314A. El objetivo vale 1 si P314A=2 y 0 si P314A=1. Predictores y objetivo se separan en el código; el split y GroupKFold usan CONGLOME. Imputación, codificación y escalamiento pertenecen al pipeline scikit-learn, ajustado dentro de cada fold.\n\nClustering compara KMeans y Agglomerative sobre muestra de 4,000 con k entre 2 y 6; PCA apoya visualización. Clasificación compara regresión logística baseline con RandomForest, ExtraTrees e HistGradientBoosting. Se usan pesos de FACTOR07 cuando existen y pesos de clase según el código.\n''',encoding='utf-8')
(OUT/'05_ARTICULO_CIENTIFICO'/'05_limitaciones_estudio.md').write_text('''# Limitaciones verificadas\n\n- El enlace individual escolar no puede hacerse: ENAHO no incluye COD_MOD/CODLOCAL. ESCALE es contexto distrital.
- Padlocal disponible carece de coordenadas numéricas/altitud; los archivos incluidos no sustentan altitud, equipamiento informático o conectividad escolar.
- `exclusion_digital` mide no uso declarado de Internet en el mes anterior, no competencia ni desempeño escolar directo.
- Los perfiles son exploratorios y dependen de muestra/variables; nombres no deben presentarse como categorías validadas.
- PR-AUC del pipeline es integral trapezoidal de la curva precisión-recall; Average Precision es una definición distinta.
- ROC/PR/métricas usan factor de expansión donde se documenta; conteos de matriz de confusión son no ponderados. No se estiman intervalos de confianza del diseño complejo.
- La definición de salida/background SHAP no queda completamente especificada para sostener interpretaciones probabilísticas.
- No se encontró evidencia exportada de una curva de calibración fuente; esta auditoría la derivó del modelo/split guardados sin reentrenar.
- La guía del docente no se localizó en el directorio del proyecto durante esta auditoría; las referencias rectoras disponibles son las instrucciones ya citadas en la conversación y matriz local de cumplimiento. Confirmar PDF fuente antes de redacción final.
''',encoding='utf-8')
(OUT/'05_ARTICULO_CIENTIFICO'/'05_conclusiones_sustentadas.md').write_text('''# Conclusiones sustentadas\n\n1. En la muestra analítica, alrededor de una cuarta parte del grupo elegible no declaró uso de Internet durante el mes previo; esto describe uso reciente, no alfabetización digital.
2. La partición por CONGLOME evita conglomerados compartidos entre entrenamiento y prueba, según auditoría persistida.
3. KMeans k=2 maximiza Silhouette entre configuraciones registradas; el contraste de perfiles sugiere diferencias rurales y de servicios, pero sigue siendo exploratorio.
4. RandomForest presenta ROC-AUC de prueba superior a 0.82 y recall cercano a 0.70, junto con precisión alrededor de 0.48 al umbral 0.5; las alertas implican falsos positivos relevantes.
5. ESCALE aporta contexto distrital, no escolar individual. No se puede concluir asociación causal ni evaluar con estos archivos altitud, equipos o conectividad escolar.
''',encoding='utf-8')
(OUT/'05_ARTICULO_CIENTIFICO'/'05_fuentes_y_referencias.md').write_text('''# Fuentes y referencias disponibles\n\n- ENAHO 2025, módulos 100/200/300, Sumaria: archivos locales descritos en `entregables/data/raw/README_fuentes.md`.
- ESCALE 2025, Matrícula, Padrón y Padlocal: artefactos locales y auditorías de cruce; usar metadatos del propio conjunto para completar cita institucional.
- Guía de evaluación del docente: el PDF se busca por nombre en la carpeta Downloads del usuario; no está dentro del workspace del proyecto en esta ejecución, por lo que no se afirma verificación directa en este paquete. No crear referencia bibliográfica final sin consultar portada/metadatos del PDF.
- No se añadieron referencias académicas externas no verificadas.
''',encoding='utf-8')
# Figures index and captions
capt=[]
for p in sorted((OUT/'GRAFICAS_PNG').glob('*.png')):
 capt.append({'archivo':p.name,'titulo':p.stem.replace('_',' '),'datos_origen':'Artefactos agregados existentes o modelo/dataset procesado documentado','algoritmo':'según nombre/README; revisar sección de origen','interpretacion':'Consultar README_PARA_CHATGPT.md y JSON de fase.','limitaciones':'No es evidencia de causalidad; calidad/resolución de origen puede variar.','ruta':str(p.relative_to(OUT))})
pd.DataFrame(capt).to_csv(OUT/'05_ARTICULO_CIENTIFICO'/'05_leyendas_figuras.csv',index=False,encoding='utf-8-sig')
# Data tables for article
for filename in ['comparacion_modelos.csv','metricas_clustering.csv','perfiles_clusters.csv','shap_importancia_top10.csv','matriz_confusion.csv','cv_groupkfold_modelos.csv']:
 cp(T/filename,OUT/'05_ARTICULO_CIENTIFICO'/('tabla_'+filename))
(OUT/'05_ARTICULO_CIENTIFICO'/'05_tablas_para_articulo.xlsx').write_text('No se generó libro Excel; usar CSV independientes adjuntos en esta carpeta.',encoding='utf-8')
# Audit inventory and rubric status
status=pd.DataFrame([
['Integración ENAHO 100/200/300/Sumaria','Parcial/verificado','Dataset, código y resumen; llaves documentadas','No se recontaron etapas originales en esta exportación'],
['ESCALE matrícula 2025 enlazada con padrón','Verificado parcial','54,038/54,038 servicios enlazados por COD_MOD+ANEXO','No permite identificar escuela ENAHO individual'],
['Clustering, dos algoritmos y k 2–6','Verificado','CSV métricas con Silhouette, DB, CH; PCA y silueta existentes','Perfiles exploratorios; faltan variables escolares ausentes'],
['PCA 2D','Verificado','Figura clusters_pca existente','Figura de origen conservada, dpi de fuente no elevado'],
['Clasificación baseline + tres modelos','Verificado','Comparación y GroupKFold guardados','Falta validación estadística por diseño muestral'],
['Test aislado por CONGLOME','Verificado','auditoria_split_conglomerado.json, cero intersecciones','Registros son evidencia del pipeline guardado'],
['ROC/PR/matriz/calibración','Parcial','ROC/PR/calibración y matriz exportadas; curvas derivadas con modelo guardado','La recalibración es una derivación de auditoría; no reentrenamiento'],
['SHAP summary, 2 dependencias, 2 casos','Parcial','Figuras y JSON existentes','Salida SHAP no declara escala explícitamente; casos anonimizados'],
['Altitud, equipos, conectividad escolar','No verificable / ausente','Limitación explícita en auditoría y reportes','No inferir ni inventar'],
['Guía original PDF del docente','Pendiente de confirmación','No localizada dentro del workspace del proyecto','No se pudo comparar exhaustivamente el paquete con el PDF fuente en esta auditoría']
],columns=['criterio','estado','evidencia','limitacion'])
status.to_csv(OUT/'00_AUDITORIA'/'00_estado_cumplimiento_rubrica.csv',index=False,encoding='utf-8-sig')
# audit source inventory
inv=[]
for p in SRC.rglob('*'):
 if p.is_file():
  inv.append({'ruta':p.relative_to(BASE).as_posix(),'bytes':p.stat().st_size,'categoria':('raw' if 'data/raw' in p.as_posix() else 'processed' if 'data/processed' in p.as_posix() else 'notebook' if p.suffix=='.ipynb' else 'script' if p.suffix=='.py' else 'output'),'incluido_en_evidencias':False if ('data/raw' in p.as_posix() or 'data/processed' in p.as_posix() or p.suffix=='.joblib' or p.name=='muestra_auditoria.csv') else True})
writej(OUT/'00_AUDITORIA'/'00_auditoria_proyecto.json',{'fecha_utc':now,'proyecto_canonico':'entregables/','inventario_fuente':inv,'conteos':{'archivos':len(inv),'scripts':sum(x['categoria']=='script' for x in inv),'notebooks':sum(x['categoria']=='notebook' for x in inv),'raw':sum(x['categoria']=='raw' for x in inv),'processed':sum(x['categoria']=='processed' for x in inv)},'notebook_execution_counts':'Inspeccionar notebook: el archivo existe; las celdas sin salida no prueban ejecución independiente. Los resultados CSV/JSON son evidencia material separada.','modelo':'outputs/models/modelo_ganador.joblib presente; utilizado solo para derivar curvas agregadas, sin volver a entrenar.','guide_pdf':'No localizado dentro de la carpeta del proyecto; se intentó la ruta Downloads proporcionada y debe confirmarse accesibilidad/ruta Unicode.','privacidad':'No se exportaron microdatos procesados, filas de muestra de auditoría, IDs personales ni probabilidades individuales.','diferencias':'Solo se consideró entregables/ como paquete fuente más reciente. Otros outputs de raíz, si existen, pueden ser copias antiguas; cotejo diferencial exhaustivo pendiente.','rubrica':status.to_dict(orient='records')})
(OUT/'00_AUDITORIA'/'00_resumen_ejecutivo.md').write_text(f'''# Resumen ejecutivo de auditoría\n\nLa carpeta fuente auditada es `entregables/`. Se localizaron pipeline Python, notebook Jupyter, base procesada, modelo RandomForest, resultados tabulares, figuras SHAP/clustering y un artículo PDF de 5 páginas. La salida analítica disponible contiene {len(df):,} filas y {df.shape[1]} columnas.\n\nLos modelos y métricas persistidas respaldan clustering KMeans k=2 (Silhouette {metric_obj['ganador']['silhouette']:.3f}) y clasificación RandomForest con ROC-AUC test guardado 0.828 y F1 macro 0.712. Las curvas ROC, PR y calibración de este paquete se derivaron de la base y modelo guardado sin entrenamiento adicional.\n\nEl test está aislado por CONGLOME (3,292 grupos train; 824 test; cero compartidos). El cruce ESCALE se documenta a nivel UBIGEO; no vincula cada estudiante a una escuela. No hay evidencia de altitud, equipamiento informático ni conectividad escolar. La definición target mide no uso mensual reportado, no destrezas digitales.\n\nLa guía PDF no se encontró dentro del workspace fuente durante esta auditoría; su disponibilidad en la ruta externa deberá comprobarse antes de afirmar cumplimiento literal. Consultar `00_estado_cumplimiento_rubrica.csv` para estados y vacíos verificables.\n''',encoding='utf-8')



