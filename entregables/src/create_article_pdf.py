"""Renderiza el artículo científico de 5 páginas a partir de resultados locales."""
from __future__ import annotations
import json, re, html
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
    TableStyle, PageBreak, KeepTogether)

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'outputs'/'tables'; F=ROOT/'outputs'/'figures'; R=ROOT/'outputs'/'reports'
OUT=R/'articulo_brecha_digital_enaho_escale_2025.pdf'

def safe(text):
    return html.escape(str(text)).replace('\n','<br/>')

def main():
    summary=json.loads((T/'resumen_ejecucion.json').read_text(encoding='utf-8'))
    models=pd.read_csv(T/'comparacion_modelos.csv').sort_values('cv_roc_auc_mean',ascending=False)
    clusters=pd.read_csv(T/'metricas_clustering.csv').sort_values(['silhouette','davies_bouldin'],ascending=[False,True])
    profiles=pd.read_csv(T/'perfiles_clusters.csv')
    top=pd.read_csv(T/'shap_importancia_top10.csv')
    locals_=json.loads((T/'shap_casos_locales.json').read_text(encoding='utf-8'))
    deps=json.loads((T/'shap_dependencias.json').read_text(encoding='utf-8'))
    missing=pd.read_csv(T/'diccionario_variables.csv')
    miss_pct=missing.set_index('variable')['faltantes_pct'].to_dict()
    best=models.iloc[0]
    best_cluster=clusters.iloc[0]
    pca_config=json.loads((T/'configuracion_clustering.json').read_text(encoding='utf-8')) if (T/'configuracion_clustering.json').exists() else {}
    pca_var=pca_config.get('varianza_pca_2d',[float('nan'),float('nan')])
    sty=getSampleStyleSheet()
    sty.add(ParagraphStyle(name='TitleX',parent=sty['Title'],fontName='Helvetica-Bold',fontSize=17,leading=20,textColor=colors.HexColor('#15324b'),alignment=TA_CENTER,spaceAfter=7))
    sty.add(ParagraphStyle(name='SubX',parent=sty['Normal'],fontName='Helvetica',fontSize=8.5,leading=11,textColor=colors.HexColor('#496477'),alignment=TA_CENTER,spaceAfter=10))
    sty.add(ParagraphStyle(name='H1X',parent=sty['Heading1'],fontName='Helvetica-Bold',fontSize=12,leading=14,textColor=colors.HexColor('#15324b'),spaceBefore=5,spaceAfter=4))
    sty.add(ParagraphStyle(name='H2X',parent=sty['Heading2'],fontName='Helvetica-Bold',fontSize=9.5,leading=11,textColor=colors.HexColor('#24627c'),spaceBefore=4,spaceAfter=3))
    sty.add(ParagraphStyle(name='BodyX',parent=sty['BodyText'],fontName='Helvetica',fontSize=8.1,leading=10.1,alignment=TA_JUSTIFY,spaceAfter=4))
    sty.add(ParagraphStyle(name='SmallX',parent=sty['BodyText'],fontName='Helvetica',fontSize=7.1,leading=8.3,alignment=TA_LEFT,spaceAfter=2))
    sty.add(ParagraphStyle(name='CaptionX',parent=sty['BodyText'],fontName='Helvetica-Oblique',fontSize=7,leading=8,textColor=colors.HexColor('#516473'),alignment=TA_CENTER,spaceBefore=2,spaceAfter=4))
    page_w,page_h=A4
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=14*mm,bottomMargin=14*mm,title='Brecha digital y educación en Perú: ENAHO-ESCALE 2025',author='Proyecto académico')
    story=[]
    def p(txt,style='BodyX'): story.append(Paragraph(txt,sty[style]))
    def h(txt): p(safe(txt),'H1X')
    def image(name,w=178*mm,hmax=75*mm):
        path=F/name
        if path.exists():
            im=Image(str(path)); scale=min(w/im.imageWidth,hmax/im.imageHeight,1); im.drawWidth*=scale; im.drawHeight*=scale; story.append(im)
    def image_pair(name1,name2,w=83*mm,hmax=48*mm):
        cells=[]
        for name in (name1,name2):
            path=F/name
            if path.exists():
                im=Image(str(path)); scale=min(w/im.imageWidth,hmax/im.imageHeight,1); im.drawWidth*=scale; im.drawHeight*=scale; cells.append(im)
            else: cells.append(Paragraph('',sty['SmallX']))
        pair=Table([[cells[0],cells[1]]],colWidths=[w,w],hAlign='CENTER')
        pair.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
        story.append(pair)
    def table(rows,widths=None,fontsize=7):
        data=[[Paragraph(safe(v),sty['SmallX']) for v in row] for row in rows]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce9f0')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#15324b')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#a8bac5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')])]))
        story.append(tab); story.append(Spacer(1,4))
    def footer(canvas,doc_):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#b8c8d1')); canvas.line(15*mm,10*mm,page_w-15*mm,10*mm)
        canvas.setFont('Helvetica',7); canvas.setFillColor(colors.HexColor('#546875'))
        canvas.drawString(15*mm,6.5*mm,'ENAHO-ESCALE 2025 · Informe académico · Semilla 42')
        canvas.drawRightString(page_w-15*mm,6.5*mm,f'{doc_.page}')
        canvas.restoreState()
    n=summary['n_filas_analiticas']; prev=100*summary['target_prevalencia']
    p('BRECHA DIGITAL Y EDUCACIÓN EN PERÚ: ANÁLISIS CON ENAHO-ESCALE 2025','TitleX')
    p('Artículo técnico · ENAHO 2025 · Validación por conglomerados · Semilla 42','SubX')
    h('Resumen')
    p(f'Se integraron los módulos ENAHO 100, 200, 300 y Sumaria 2025 con el padrón ESCALE y matrícula 2025 disponibles en el proyecto. La población analítica comprende {n:,} escolares asistentes de 6 a 17 años con respuesta válida sobre uso de Internet. El resultado es no haber usado Internet el mes anterior (prevalencia {prev:.1f}%). Se compararon dos algoritmos de clustering y cuatro clasificadores con separación por CONGLOME. {best.modelo} obtuvo ROC-AUC ponderado {best.test_roc_auc:.3f}, PR-AUC {best.test_pr_auc:.3f} y F1 macro {best.test_f1_macro:.3f} en test agrupado. La interpretación se limita al uso reciente de Internet; las medidas ESCALE son contexto distrital y no identifican la escuela del estudiante.')
    p('<b>Palabras clave:</b> brecha digital; educación; ENAHO; ESCALE; agrupamiento; clasificación; SHAP.','SmallX')
    h('1. Introducción')
    p('La brecha digital escolar combina condiciones del hogar, uso de tecnologías y oferta educativa local. Este análisis responde a una pregunta predictiva acotada: entre escolares asistentes de 6 a 17 años, ¿qué características ayudan a ordenar a quienes no reportaron uso de Internet en el mes previo? La respuesta es una asociación del modelo, no una estimación causal ni una medición de aprendizaje.')
    h('Fuentes y cruce')
    mat_audit=json.loads((T/'auditoria_cruce_matricula_2025.json').read_text(encoding='utf-8')) if (T/'auditoria_cruce_matricula_2025.json').exists() else {}
    p(f"Las llaves personales se conservaron como texto. ENAHO 200 y 300 se enlazaron por identificadores de persona; los módulos 100 y Sumaria por llaves del hogar. Padron y matrícula escolar se vincularon por COD_MOD+ANEXO y se agregaron por UBIGEO. La auditoría registra {mat_audit.get('servicios_enlazados_al_padron','no disponible')}/{mat_audit.get('servicios_matricula_cod_mod_anexo','no disponible')} servicios enlazados en los archivos disponibles. Padlocal se usó para contar locales por UBIGEO y revisar CODLOCAL. ENAHO no tiene COD_MOD/CODLOCAL, por lo que el indicador ESCALE es territorial, no individual.")
    image('diagrama_cruce_tablas.png',178*mm,31*mm); p('Figura 1. Llaves de integración y límite del enlace escolar.','CaptionX')
    story.append(PageBreak())

    h('2. Datos y metodología')
    p('El objetivo vale 1 para P314A=2 y 0 para P314A=1, entre personas de 6 a 17 años con P307=1. Se conservaron edad, grado, sexo, lengua materna, características del hogar, geografía y medidas distritales disponibles. P1121 se interpreta como electricidad del hogar, no como conectividad; P1144 identifica Internet doméstico.')
    h('Variables TIC y calidad de datos')
    p('Se incorporaron P314B (lugar), P314D (frecuencia), P316 (propósito), P316B (uso de computadora/tableta) y P316C1-C10 (tareas digitales declaradas). Los lugares, frecuencia y propósitos se describen, pero no se usan como predictores porque dependen de haber usado Internet y podrían filtrar el resultado. P316B y el conteo de tareas P316C entran al análisis predictivo con imputación y señalización de faltantes dentro de cada fold. Las tareas son una aproximación autodeclarada, no una prueba de destreza.')
    p('Los porcentajes de faltantes por variable están en el diccionario analítico; N válido y faltante por ítem TIC están en la tabla de frecuencias. El tratamiento no aprende parámetros con el test: imputación, escalado y one-hot se ajustan dentro del pipeline de cada fold.')
    missing_focus=['P314A','P314D','tic_frecuencia_uso_cluster','rezago_escolar_anios','agua_potable','saneamiento_red_publica','P316B','tic_habilidades_digitales_n','tic_uso_computadora_ult3m','P1144','esc_ratio_alumno_docente','esc_matricula_primaria_2025','esc_n_locales_padlocal']
    missrows=missing[missing.variable.isin(missing_focus)].sort_values('variable')
    table([['Variable analítica','Faltantes (%)']]+[[r.variable,f'{r.faltantes_pct:.1f}'] for r in missrows.itertuples(index=False)],[100*mm,35*mm])
    p('P316B y el conteo de tareas tienen faltantes altos por saltos y preguntas condicionadas. No se convierten a "no"; la imputación con indicadores limita la fuga entre folds, pero el modelo puede aprender el patrón de respuesta y por eso estas asociaciones se interpretan con cautela.')
    h('Integración ESCALE')
    p('Padlocal.dbf se utilizó para contar locales por UBIGEO y comprobar coincidencia de CODLOCAL con Padron. No tiene coordenadas numéricas ni altitud. El Padron aporta agregados disponibles de matrícula, docentes, gestión y coordenadas. Los archivos locales no permiten derivar altitud, equipamiento informático o conectividad escolar; no se imputan ni infieren. Sin COD_MOD/CODLOCAL en ENAHO, no se atribuye la escuela concreta a cada menor.')
    h('Clustering')
    p(f'Se tomó una muestra reproducible de hasta 4,000 filas, con imputación, escalado y codificación, y se compararon K-Means y Agglomerative para k=2 a 6 sobre 20 componentes principales. UBIGEO se excluyó como identificador del clustering. La solución con mayor silueta fue {best_cluster.algoritmo}, k={int(best_cluster.k)} (silueta={best_cluster.silhouette:.3f}; Davies-Bouldin={best_cluster.davies_bouldin:.3f}; Calinski-Harabasz={best_cluster.calinski_harabasz:.1f}). En la proyección PCA, los dos ejes explican {100*pca_var[0]:.1f}% y {100*pca_var[1]:.1f}% de varianza respectivamente. Los perfiles incluyen ingreso y gasto per cápita, ratio alumnos/docente distrital, frecuencia TIC P314D (0=no uso), años de rezago y servicios básicos. P314D se limita al clustering para evitar fuga en clasificación. Los nombres se basan en ruralidad y no uso relativo, con medias detalladas abajo; son descriptivos y exploratorios.')
    image('silhouette_clusters.png',160*mm,60*mm); p('Figura 2. Coeficientes de silueta de la solución seleccionada.','CaptionX')
    story.append(PageBreak())

    h('3. Resultados de perfiles y clasificación')
    rows=[['Algoritmo','k','Silueta','Davies-B.','Calinski-H.']]
    rows += [[r.algoritmo,int(r.k),f'{r.silhouette:.3f}',f'{r.davies_bouldin:.3f}',f'{r.calinski_harabasz:.1f}'] for r in clusters.itertuples(index=False)]
    table(rows,[43*mm,12*mm,24*mm,30*mm,32*mm])
    prows=[['Perfil','n','No uso','Frecuencia TIC','Ratio A/D','Ingreso pc (miles)','Rezago (años)','Agua potable','Saneam. red']]
    for r in profiles.itertuples(index=False):
        prows.append([r.nombre_perfil,int(r.n),f'{100*r.exclusion_digital:.1f}%',f'{r.tic_frecuencia_uso_cluster:.2f}',f'{r.esc_ratio_alumno_docente:.1f}',f'{r.ingreso_hogar_percapita/1000:.1f}',f'{r.rezago_escolar_anios:.2f}',f'{r.agua_potable:.2f}',f'{r.saneamiento_red_publica:.2f}'])
    table(prows,[42*mm,10*mm,16*mm,22*mm,17*mm,20*mm,17*mm,17*mm,17*mm])
    sizes=', '.join(f"{r.nombre_perfil}: n={int(r.n)}" for r in profiles.itertuples(index=False))
    p(f'La distribución observada es {sizes}. La tabla permite comprobar las diferencias de ruralidad, Internet del hogar, no uso, habilidades declaradas, rezago y acceso a servicios. La silueta orienta la separación geométrica; no valida por sí sola utilidad educativa ni convierte los nombres exploratorios en categorías oficiales.')
    image('clusters_pca.png',146*mm,51*mm); p('Figura 3. Proyección PCA bidimensional; solo visualización.','CaptionX')
    h('Clasificación y prevención de fuga')
    p('GroupShuffleSplit aisló 20% de conglomerados para prueba; la auditoría confirma que no hay conglomerados compartidos. GroupKFold comparó candidatos en entrenamiento y guarda ROC-AUC, PR-AUC y F1 macro ponderados por fold en el archivo suplementario. Se excluyeron P307 y P314A de predictores. P314B, P314D y P316 no entran al clasificador supervisado por su condicionamiento al uso; P314D se usa solo en clustering. P316B y P316C sí se incluyen como predictores observados, con faltantes tratados dentro de cada fold. Se evaluaron una regresión logística base, Random Forest, Extra Trees e HistGradientBoosting; class_weight atendió el desbalance y FACTOR07 ponderó ajuste y métricas cuando estuvo disponible.')
    story.append(PageBreak())

    h('4. Desempeño predictivo')
    rows=[['Modelo','ROC CV','DE ROC','PR CV','F1 CV','ROC test','PR test','F1 test','Precisión','Recall']]
    for r in models.itertuples(index=False): rows.append([r.modelo.replace('_baseline',''),f'{r.cv_roc_auc_mean:.3f}',f'{r.cv_roc_auc_sd:.3f}',f'{r.cv_pr_auc_mean:.3f}',f'{r.cv_f1_macro_mean:.3f}',f'{r.test_roc_auc:.3f}',f'{r.test_pr_auc:.3f}',f'{r.test_f1_macro:.3f}',f'{r.test_precision:.3f}',f'{r.test_recall:.3f}'])
    table(rows,[29*mm,17*mm,13*mm,16*mm,14*mm,17*mm,16*mm,15*mm,17*mm,16*mm])
    p(f'{best.modelo} obtuvo el mejor ROC-AUC medio de validación por conglomerado ({best.cv_roc_auc_mean:.3f}; DE {best.cv_roc_auc_sd:.3f}), con PR-AUC CV={best.cv_pr_auc_mean:.3f} y F1 macro CV={best.cv_f1_macro_mean:.3f}. En el test aislado, ROC-AUC={best.test_roc_auc:.3f}, PR-AUC={best.test_pr_auc:.3f} y Brier ponderado={best.test_brier:.3f}, frente a prevalencia de {prev:.1f}%. Con umbral 0.5, precisión={best.test_precision:.3f} y recall={best.test_recall:.3f}; este umbral no representa una decisión de política calibrada a costos. El desglose por fold está en cv_groupkfold_modelos.csv; la auditoría de separación y la distribución de clases están en las tablas suplementarias.')
    cm=pd.read_csv(T/'matriz_confusion.csv',index_col=0).to_numpy(); tn,fp,fn,tp=cm.ravel()
    p(f'La matriz de conteos registra TN={tn:,}, FP={fp:,}, FN={fn:,} y TP={tp:,}. Los falsos negativos son casos de no uso no detectados; los falsos positivos priorizarían casos que sí reportaron uso. La tabla ponderada también está disponible para distinguir conteos de escala poblacional.')
    image('evaluacion_modelos.png',178*mm,68*mm); p('Figura 4. Curvas ROC, precisión–recall y calibración del mejor modelo en test agrupado.','CaptionX')
    h('Importancia global SHAP')
    top_rows=[['Variable fuente','Media |SHAP|']]+[[r.variable,f'{r.mean_abs_shap:.4f}'] for r in top.head(10).itertuples(index=False)]
    table(top_rows,[98*mm,35*mm])
    p(f"{top.iloc[0].variable.replace('num__','').replace('cat__','')} y {top.iloc[1].variable.replace('num__','').replace('cat__','')} encabezan la media absoluta SHAP en esta ejecución. La importancia mide magnitud media de contribución del modelo, no dirección causal. Las variables ESCALE de coordenadas describen patrones geográficos y no deben interpretarse como efectos propios de la escuela del escolar.")
    image('shap_beeswarm.png',160*mm,55*mm); p('Figura 5. Beeswarm SHAP por variable original: dirección y dispersión de las contribuciones.','CaptionX')
    story.append(PageBreak())

    h('5. Dependencias, casos locales y discusión')
    p('Las dependencias seleccionadas muestran P1144 en unidades originales con la lengua materna ENAHO (Quechua, Aimara y otras/faltante), y el ratio distrital de alumnos/docente coloreado por Internet en el hogar. La dispersión representa contribuciones SHAP del predictor Internet en el hogar condicionadas a los otros predictores, y no demuestra que modificar conexión o ratio cause cambios en el resultado.')
    image_pair('shap_dependence_1.png','shap_dependence_2.png')
    p('Figura 6. Dependencia de P1144: interacción exploratoria con lengua materna y ratio distrital.','CaptionX')
    high=next((c for c in locals_ if c['caso']=='alto'),None); low=next((c for c in locals_ if c['caso']=='bajo'),None)
    def local_vars(case,sign):
        return ', '.join(f"{x['variable'].replace('num__','').replace('cat__','')} ({x['shap']:+.3f})" for x in case['contribuciones_top'] if (x['shap']>0 if sign=='positiva' else x['shap']<0)) if case else 'sin artefacto local'
    p(f"Los waterfall describen extremos correctamente clasificados de todo el test por {best.modelo}. Caso alto (probabilidad {high['probabilidad']:.3f}): elevaron la predicción {local_vars(high,'positiva')}; la redujeron {local_vars(high,'negativa')}. Caso bajo (probabilidad {low['probabilidad']:.3f}): elevaron la salida {local_vars(low,'positiva')}; la redujeron {local_vars(low,'negativa')}. Son explicaciones de dos observaciones.")
    image_pair('shap_waterfall_alto.png','shap_waterfall_bajo.png',83*mm,49*mm)
    p('Figura 7. Waterfall: caso correctamente clasificado con mayor y menor probabilidad predicha entre los explicados.','CaptionX')
    h('Conclusiones y limitaciones')
    p('El pipeline produce una comparación predictiva reproducible con separación por conglomerados y evidencia explícita de las variables usadas y excluidas. El uso mensual de Internet es una medida concreta, más estrecha que conectividad estable, destreza digital o aprendizaje. Las tareas P316C permiten describir acciones autodeclaradas. ESCALE agrega contexto por UBIGEO: el Padlocal existente no ofrece coordenadas numéricas y la llave institucional no aparece en ENAHO. Los datos disponibles tampoco sustentan altitud, equipamiento o conectividad escolar; estos aspectos deben presentarse como limitaciones.')
    p('Se recomienda usar las estimaciones para orientar diagnóstico territorial, validar perfiles con actores locales y no asignar recursos individuales automáticamente. Una evaluación externa y mediciones escolares con llave compatible serían necesarias para extender las conclusiones.','BodyX')
    p('<b>Referencias:</b> INEI, <i>Diccionario de Datos ENAHO 2025</i>, https://proyectos.inei.gob.pe/iinei/srienaho/Descarga/DocumentosMetodologicos/2025-55/20_Diccionario_2025.pdf; MINEDU, <i>Catálogo Censo Educativo 2025</i>, https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985; Lundberg y Lee (2017), SHAP; Pedregosa et al. (2011), scikit-learn.','SmallX')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__=='__main__': main()


