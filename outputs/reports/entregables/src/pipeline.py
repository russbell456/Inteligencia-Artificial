"""Construcción de datos, clustering, clasificación agrupada y explicabilidad.

Run from the project root with: python -m src.pipeline
"""
from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dbfread import DBF
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (auc, calinski_harabasz_score, classification_report,
    confusion_matrix, davies_bouldin_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
    silhouette_samples)
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
if not (RAW / "Enaho01-2025-100.csv").exists():
    RAW = ROOT
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "outputs" / "models"
REPORTS = ROOT / "outputs" / "reports"
for folder in (PROCESSED, TABLES, FIGURES, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("enaho_escale")
KEYS_HH = ["AÑO", "MES", "CONGLOME", "VIVIENDA", "HOGAR", "UBIGEO"]
KEYS_PERSON = KEYS_HH + ["CODPERSO"]
NULLS = ["", " ", "  ", "   ", "NA", "N/A"]


def read_inei(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Lee CSV INEI preservando códigos y normalizando espacios como nulos."""
    df = pd.read_csv(path, sep=";", encoding="latin1", dtype=str, na_values=NULLS,
                     keep_default_na=True, low_memory=False, usecols=usecols)
    df.columns = df.columns.str.upper().str.strip()
    # El Latin-1 del archivo contiene AÑO. Esta sustitución atiende archivos con encabezado deformado.
    df = df.rename(columns={c: "AÑO" for c in df.columns if c.replace("�", "Ñ") == "AÑO"})
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().replace("", np.nan)
    return df


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """Convierte en sitio las columnas ENAHO que usan coma decimal."""
    for col in columns:
        if col in df:
            df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")


def load_escale() -> pd.DataFrame:
    """Agrega el padrón escolar por UBIGEO y ámbito censal disponible."""
    path = RAW / "Padron.dbf"
    if not path.exists():
        LOG.warning("Padron.dbf no encontrado; se continúa sin indicadores ESCALE.")
        return pd.DataFrame(columns=["UBIGEO", "AREA_CENSO", "esc_n_escuelas", "esc_alumnos",
                                     "esc_docentes", "esc_secciones", "esc_ratio_alumno_docente",
                                     "esc_prop_gestion_publica", "esc_lat_media", "esc_lon_media"])
    fields = ["CODGEO", "DAREACENSO", "GESTION", "TALUM", "TDOC", "TSEC", "NLAT_IE", "NLONG_IE"]
    raw = pd.DataFrame(iter(DBF(str(path), encoding="latin1", char_decode_errors="replace")))
    raw.columns = raw.columns.str.upper().str.strip()
    for field in fields:
        if field not in raw:
            raw[field] = np.nan
    raw["UBIGEO"] = raw["CODGEO"].astype("string").str.strip().str.zfill(6)
    for col in ["TALUM", "TDOC", "TSEC", "NLAT_IE", "NLONG_IE"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    # GES_DEP expresa dependencia administrativa (pública/privada); se conserva la distribución
    # observada en vez de inventar una recodificación si cambian las categorías del padrón.
    # D_GESTION indica Pública/Privada; D_GES_DEP identifica el sector funcional
    # (por ejemplo, Sector Educación) y no debe usarse como tipo de gestión.
    admin_label = raw.get("D_GESTION", pd.Series("", index=raw.index)).astype("string")
    raw["_publica"] = admin_label.str.strip().str.upper().str.contains("PÚBLIC|PUBLIC|ESTATAL", na=False).astype(float)
    group = raw.groupby(["UBIGEO", "DAREACENSO"], dropna=False)
    out = group.agg(esc_n_escuelas=("COD_MOD", "nunique"), esc_alumnos=("TALUM", "sum"),
                    esc_docentes=("TDOC", "sum"), esc_secciones=("TSEC", "sum"),
                    esc_prop_gestion_publica=("_publica", "mean"), esc_lat_media=("NLAT_IE", "mean"),
                    esc_lon_media=("NLONG_IE", "mean")).reset_index()
    out["esc_ratio_alumno_docente"] = out["esc_alumnos"] / out["esc_docentes"].replace(0, np.nan)
    out = out.rename(columns={"DAREACENSO": "AREA_CENSO"})
    LOG.info("ESCALE agregado: %s grupos distrito-área", f"{len(out):,}")
    return out


def load_escale_padlocal() -> pd.DataFrame:
    """Usa Padlocal para contar locales por UBIGEO y auditar CODLOCAL vs Padron."""
    path = RAW / "Padlocal.dbf"
    columns = ["UBIGEO", "esc_n_locales_padlocal"]
    if not path.exists():
        LOG.warning("Padlocal.dbf no encontrado; se continúa sin el recuento de locales.")
        return pd.DataFrame(columns=columns)
    local = pd.DataFrame(iter(DBF(str(path), encoding="latin1", char_decode_errors="replace")))
    local.columns = local.columns.str.upper().str.strip()
    for col in ["CODLOCAL", "CODGEO"]:
        if col not in local:
            raise ValueError(f"Padlocal.dbf no contiene {col}.")
    local["CODLOCAL"] = local["CODLOCAL"].astype("string").str.strip().str.zfill(6)
    local["UBIGEO"] = local["CODGEO"].astype("string").str.strip().str.zfill(6)
    local = local.dropna(subset=["UBIGEO", "CODLOCAL"]).drop_duplicates("CODLOCAL")
    result = local.groupby("UBIGEO", as_index=False).agg(esc_n_locales_padlocal=("CODLOCAL", "nunique"))
    pad_path = RAW / "Padron.dbf"
    matched = 0
    padron_locales = 0
    ubigeo_agreements = 0
    if pad_path.exists():
        pad = pd.DataFrame(iter(DBF(str(pad_path), encoding="latin1", char_decode_errors="replace")))
        pad.columns = pad.columns.str.upper().str.strip()
        if {"CODLOCAL", "CODGEO"}.issubset(pad.columns):
            pad = pad[["CODLOCAL", "CODGEO"]].dropna().copy()
            pad["CODLOCAL"] = pad["CODLOCAL"].astype("string").str.strip().str.zfill(6)
            pad["UBIGEO_PADRON"] = pad["CODGEO"].astype("string").str.strip().str.zfill(6)
            pad = pad.drop_duplicates("CODLOCAL")
            padron_locales = len(pad)
            check = pad.merge(local[["CODLOCAL", "UBIGEO"]], on="CODLOCAL", how="inner", validate="one_to_one")
            matched = len(check)
            ubigeo_agreements = int(check["UBIGEO_PADRON"].eq(check["UBIGEO"]).sum())
    audit = {
        "archivo": path.name,
        "campos_disponibles": list(local.columns),
        "locales_unicos_padlocal": int(local["CODLOCAL"].nunique()),
        "locales_unicos_padron": int(padron_locales),
        "locales_comunes_por_codlocal": int(matched),
        "ubigeo_coincidente_entre_archivos": int(ubigeo_agreements),
        "campos_de_coordenadas_o_altitud_en_padlocal": [c for c in ["NLAT_IE", "NLONG_IE", "ALTITUD", "ALTITUD_IE"] if c in local.columns],
        "nota": "Padlocal aporta CODLOCAL y geografía administrativa, pero no columnas numéricas de coordenadas/altitud; Padron aporta coordenadas cuando están informadas."
    }
    (TABLES / "auditoria_cruce_padlocal.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_escale_matricula_2025() -> pd.DataFrame:
    """Enlaza matrícula del Censo 2025 con el padrón por COD_MOD+ANEXO y agrega por distrito."""
    path = RAW / "Matricula_01_2025.dbf"
    if not path.exists() and (ROOT / "data" / "raw" / "Matricula_01_2025.dbf").exists():
        path = ROOT / "data" / "raw" / "Matricula_01_2025.dbf"
    columns = ["UBIGEO", "esc_matricula_primaria_2025", "esc_matricula_secundaria_2025"]
    if not path.exists():
        LOG.warning("Matricula_01_2025.dbf no encontrado; se omiten sus indicadores.")
        return pd.DataFrame(columns=columns)

    mat = pd.DataFrame(iter(DBF(str(path), encoding="latin1", char_decode_errors="replace")))
    mat.columns = mat.columns.str.upper().str.strip()
    dcols = [f"D{i:02d}" for i in range(1, 13)]
    required = ["COD_MOD", "ANEXO", "NROCED", "CUADRO", *dcols]
    missing = [c for c in required if c not in mat]
    if missing:
        raise ValueError(f"Campos faltantes en Matricula_01_2025.dbf: {missing}")

    # C201: matrícula por grado/sexo/turno. 3AP=primaria y 3AS=secundaria.
    mat = mat[(mat["CUADRO"].astype("string").str.strip().str.upper() == "C201") &
              mat["NROCED"].astype("string").str.strip().str.upper().isin(["3AP", "3AS"])].copy()
    if mat.empty:
        LOG.warning("Sin registros C201 para primaria/secundaria en Matricula_01_2025.dbf.")
        return pd.DataFrame(columns=columns)
    mat["COD_MOD"] = mat["COD_MOD"].astype("string").str.strip().str.zfill(7)
    mat["ANEXO"] = mat["ANEXO"].fillna("").astype("string").str.strip()
    for col in dcols:
        mat[col] = pd.to_numeric(mat[col], errors="coerce").fillna(0)
    mat["_total"] = mat[dcols].sum(axis=1)
    mat["_primaria"] = np.where(mat["NROCED"].astype("string").str.strip().str.upper() == "3AP", mat["_total"], 0)
    mat["_secundaria"] = np.where(mat["NROCED"].astype("string").str.strip().str.upper() == "3AS", mat["_total"], 0)
    service = mat.groupby(["COD_MOD", "ANEXO"], as_index=False).agg(
        esc_matricula_primaria_2025=("_primaria", "sum"),
        esc_matricula_secundaria_2025=("_secundaria", "sum"))

    pad_path = RAW / "Padron.dbf"
    if not pad_path.exists() and (ROOT / "data" / "raw" / "Padron.dbf").exists():
        pad_path = ROOT / "data" / "raw" / "Padron.dbf"
    pad = pd.DataFrame(iter(DBF(str(pad_path), encoding="latin1", char_decode_errors="replace")))
    pad.columns = pad.columns.str.upper().str.strip()
    pad = pad[["COD_MOD", "ANEXO", "CODGEO"]].dropna(subset=["COD_MOD", "CODGEO"]).copy()
    pad["COD_MOD"] = pad["COD_MOD"].astype("string").str.strip().str.zfill(7)
    pad["ANEXO"] = pad["ANEXO"].fillna("").astype("string").str.strip()
    pad["UBIGEO"] = pad["CODGEO"].astype("string").str.strip().str.zfill(6)
    pad = pad.groupby(["COD_MOD", "ANEXO"], as_index=False).agg(UBIGEO=("UBIGEO", "first"))
    joined = service.merge(pad, on=["COD_MOD", "ANEXO"], how="left", validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    coverage = {
        "archivo": path.name,
        "registros_filtrados_c201_3ap_3as": int(len(mat)),
        "servicios_matricula_cod_mod_anexo": int(len(service)),
        "servicios_enlazados_al_padron": int(matched.sum()),
        "servicios_sin_coincidencia": int((~matched).sum()),
        "proporcion_servicios_enlazados": float(matched.mean()) if len(joined) else 0.0,
        "llave_matricula_padron": ["COD_MOD", "ANEXO"],
        "llave_para_enaho": "No existe COD_MOD/CODLOCAL en los módulos ENAHO 2025 disponibles; agregación posterior por UBIGEO.",
    }
    (TABLES / "auditoria_cruce_matricula_2025.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    district = joined.loc[matched].groupby("UBIGEO", as_index=False).agg(
        esc_matricula_primaria_2025=("esc_matricula_primaria_2025", "sum"),
        esc_matricula_secundaria_2025=("esc_matricula_secundaria_2025", "sum"))
    district.to_csv(TABLES / "matricula_escale_distrito_2025.csv", index=False, encoding="utf-8")
    LOG.info("Matrícula 2025: %s/%s servicios vinculados a Padron.dbf por COD_MOD+ANEXO; se agrega a UBIGEO.",
             int(matched.sum()), len(joined))
    return district


def save_tic_description(df: pd.DataFrame) -> None:
    """Produce frecuencias reproducibles de las variables TIC solicitadas."""
    rows=[]
    weight=pd.to_numeric(df.get("factor_expansion",pd.Series(np.nan,index=df.index)),errors="coerce")
    for col in ["P314D","P316B",*[f"P314B${i}" for i in range(1,8)],*[f"P316${i}" for i in range(1,13)],*[f"P316C{i}" for i in range(1,11)]]:
        if col not in df: continue
        values=pd.to_numeric(df[col],errors="coerce")
        valid=values.notna()
        denom=float(weight[valid].sum()) if weight[valid].notna().any() else float(valid.sum())
        for code,n in values[valid].value_counts().sort_index().items():
            use_weight=weight[valid & values.eq(code)]
            numerator=float(use_weight.sum()) if use_weight.notna().any() else float((valid & values.eq(code)).sum())
            rows.append({"variable":col,"codigo":int(code),"n":int((valid & values.eq(code)).sum()),
                         "porcentaje_valido":100*n/valid.sum() if valid.sum() else np.nan,
                         "porcentaje_ponderado_valido":100*numerator/denom if denom else np.nan,
                         "n_valido":int(valid.sum()),"n_faltante":int((~valid).sum())})
    pd.DataFrame(rows).to_csv(TABLES/"resumen_variables_tic_enaho.csv",index=False,encoding="utf-8")


def save_integration_diagram() -> None:
    """Gráfico del cruce y llaves realmente usadas."""
    fig,ax=plt.subplots(figsize=(11,3.5)); ax.set_axis_off()
    boxes=[(0.02,0.35,0.19,0.38,"ENAHO 200 + 300\nPersona\nllaves + CODPERSO"),
           (0.28,0.35,0.19,0.38,"ENAHO 100 + Sumaria\nHogar\nllaves sin CODPERSO"),
           (0.55,0.35,0.19,0.38,"ESCALE Padron +\nMatrícula 2025\nCOD_MOD + ANEXO"),
           (0.81,0.35,0.17,0.38,"Tabla escolar\ncontexto distrito\nUBIGEO")]
    for x,y,w,h,label in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#edf4fb",edgecolor="#214761",linewidth=1.5,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9,transform=ax.transAxes)
    for x1,x2 in [(0.21,0.28),(0.47,0.55),(0.74,0.81)]:
        ax.annotate("",xy=(x2,0.54),xytext=(x1,0.54),xycoords=ax.transAxes,textcoords=ax.transAxes,arrowprops={"arrowstyle":"->","color":"#214761","lw":1.5})
    ax.text(.50,.12,"ENAHO no trae COD_MOD/CODLOCAL: ESCALE no identifica la escuela de cada estudiante",ha="center",va="center",fontsize=9,color="#7b3228",transform=ax.transAxes)
    fig.savefig(FIGURES/"diagrama_cruce_tablas.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def build_dataset() -> pd.DataFrame:
    """Cruza módulos ENAHO 100/200/300/Sumaria y contexto ESCALE por claves."""
    p100 = RAW / "Enaho01-2025-100.csv"
    p200 = RAW / "Enaho01-2025-200.csv"
    p300 = RAW / "Enaho01A-2025-300.csv"
    psum = RAW / "Sumaria-2025.csv"
    for path in (p100, p200, p300, psum):
        if not path.exists():
            raise FileNotFoundError(f"Falta archivo ENAHO requerido: {path.name}")
    hh100 = read_inei(p100)
    people = read_inei(p200)
    edu = read_inei(p300)
    summ = read_inei(psum)
    # Conserva únicamente variables necesarias para reducir memoria; las llaves son texto.
    hh_cols = KEYS_HH + [c for c in ["DOMINIO", "ESTRATO", "P110", "P110A1", "P111A", "P1121", "P1144", "FACTOR07"] if c in hh100]
    hh100 = hh100[hh_cols].drop_duplicates(KEYS_HH, keep="last")
    summ_cols = KEYS_HH + [c for c in ["MIEPERHO", "TOTMIEHO", "GASHOG2D", "INGHOG2D", "POBREZA", "FACTOR07"] if c in summ]
    summ = summ[summ_cols].drop_duplicates(KEYS_HH, keep="last")
    person_cols = KEYS_PERSON + [c for c in ["P203", "P207", "P208A"] if c in people]
    people = people[person_cols]
    numeric(people, ["P208A"])
    people = people[people["P208A"].between(6, 17)].copy()
    tic_place_cols = [f"P314B${i}" for i in range(1, 8)]
    tic_purpose_cols = [f"P316${i}" for i in range(1, 13)]
    tic_skill_cols = [f"P316C{i}" for i in range(1, 11)]
    tic_cols = ["P314D", "P316B", *tic_place_cols, *tic_purpose_cols, *tic_skill_cols]
    edu_cols = KEYS_PERSON + [c for c in ["P300A", "P306", "P307", "P308A", "P308B", "P308C", "P314A", "FACTOR07", *tic_cols] if c in edu]
    edu = edu[edu_cols].drop_duplicates(KEYS_PERSON, keep="last")
    df = people.merge(edu, on=KEYS_PERSON, how="inner", validate="one_to_one", suffixes=("", "_edu"))
    df = df.merge(hh100.drop(columns=["FACTOR07"], errors="ignore"), on=KEYS_HH,
                  how="left", validate="many_to_one")
    df = df.merge(summ.drop(columns=["FACTOR07"], errors="ignore"), on=KEYS_HH, how="left",
                  validate="many_to_one", suffixes=("", "_sum"))
    # Padrón escolar: las categorías del ámbito de ENAHO y ESCALE no siempre son equivalentes;
    # se integra por UBIGEO para un contexto distrital y se reporta explícitamente la limitación.
    esc = load_escale()
    if len(esc):
        esc_dist = esc.groupby("UBIGEO", as_index=False).agg(
            esc_n_escuelas=("esc_n_escuelas", "sum"), esc_alumnos=("esc_alumnos", "sum"),
            esc_docentes=("esc_docentes", "sum"), esc_secciones=("esc_secciones", "sum"),
            esc_prop_gestion_publica=("esc_prop_gestion_publica", "mean"),
            esc_lat_media=("esc_lat_media", "mean"), esc_lon_media=("esc_lon_media", "mean"),
            esc_ratio_alumno_docente=("esc_ratio_alumno_docente", "mean"))
        matricula_dist = load_escale_matricula_2025()
        if len(matricula_dist):
            esc_dist = esc_dist.merge(matricula_dist, on="UBIGEO", how="left", validate="one_to_one")
        local_dist = load_escale_padlocal()
        if len(local_dist):
            esc_dist = esc_dist.merge(local_dist, on="UBIGEO", how="left", validate="one_to_one")
        df = df.merge(esc_dist, on="UBIGEO", how="left", validate="many_to_one")
    numeric(df, ["P207", "P203", "P300A", "P306", "P307", "P308A", "P308B", "P308C", "P314A",
                 "P110", "P110A1", "P111A", "P1121", "P1144", "MIEPERHO", "TOTMIEHO", "GASHOG2D", "INGHOG2D", "POBREZA", "FACTOR07", *tic_cols])
    df["edad"] = df["P208A"]
    df["internet_hogar"] = df["P1144"]
    # El target es operacional: escolar que reporta asistir y no usó Internet el mes previo.
    # No se usa P307 ni P314A como predictor; se excluyen no asistentes/no informados.
    df = df[(df["P307"] == 1) & df["P314A"].isin([1, 2])].copy()
    df["exclusion_digital"] = (df["P314A"] == 2).astype("int8")
    # Retraso frente a la edad normativa: primaria grado 1=6 años; secundaria 1=12 años.
    grade_primary = df["P308C"].where(df["P308A"].eq(2) & df["P308C"].between(1, 6))
    grade_secondary = df["P308B"].where(df["P308A"].eq(3) & df["P308B"].between(1, 5))
    df["grado_ebr"] = grade_primary.fillna(grade_secondary)
    df["edad_normativa"] = np.select(
        [grade_primary.notna(), grade_secondary.notna()],
        [grade_primary + 5, grade_secondary + 11], default=np.nan).astype(float)
    df["rezago_escolar_anios"] = (df["edad"] - df["edad_normativa"]).clip(lower=0)
    df.loc[df["edad_normativa"].isna(), "rezago_escolar_anios"] = np.nan
    df["rezago_escolar"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    valid_grade = df["edad_normativa"].notna()
    df.loc[valid_grade, "rezago_escolar"] = (df.loc[valid_grade, "edad"] > df.loc[valid_grade, "edad_normativa"]).astype("int8")
    # Servicio básico del hogar: agua potable, saneamiento público y electricidad.
    if "P110A1" in df:
        df["agua_potable"] = df["P110A1"].map({1: 1, 2: 0})
    if "P110" in df:
        df["agua_red_publica"] = np.where(df["P110"].notna(), df["P110"].isin([1, 2]).astype(int), np.nan)
    if "P111A" in df:
        df["saneamiento_red_publica"] = np.where(df["P111A"].notna(), df["P111A"].isin([1, 2]).astype(int), np.nan)
    # Para perfiles, 0=sin uso reportado; 1-4 siguen las categorías de frecuencia P314D.
    df["tic_frecuencia_uso_cluster"] = df["P314D"]
    df.loc[df["P314A"].eq(2), "tic_frecuencia_uso_cluster"] = 0
    # ENAHO TIC: lugar/frecuencia/propósito se describen y se excluyen de predictores;
    # P316B y P316C se conservan para el análisis predictivo con faltantes explícitos.
    for i in range(1, 8):
        source = f"P314B${i}"
        if source in df:
            df[f"tic_lugar_{i}"] = (df[source] == i).astype("Int8")
    for i in range(1, 13):
        source = f"P316${i}"
        if source in df:
            df[f"tic_proposito_{i}"] = (df[source] == 1).astype("Int8")
    skill_cols = [c for c in [f"P316C{i}" for i in range(1, 11)] if c in df]
    if skill_cols:
        answered_skills = df[skill_cols].notna().sum(axis=1)
        df["tic_habilidades_digitales_n"] = (df[skill_cols] == 1).sum(axis=1).where(answered_skills.gt(0))
    if "P316B" in df:
        df["tic_uso_computadora_ult3m"] = df["P316B"].map({1: 1, 2: 0})
    df["ingreso_hogar_percapita"] = df.get("INGHOG2D", np.nan) / df.get("MIEPERHO", np.nan).replace(0, np.nan)
    df["gasto_hogar_percapita"] = df.get("GASHOG2D", np.nan) / df.get("MIEPERHO", np.nan).replace(0, np.nan)
    df["rural"] = df["ESTRATO"].astype("string").isin(["6", "7", "8"]).astype("int8") if "ESTRATO" in df else np.nan
    df["grupo_conglomerado"] = df["CONGLOME"].astype("string")
    df["factor_expansion"] = df["FACTOR07"].fillna(df.get("FACTOR07_sum", np.nan))
    save_tic_description(df)
    out = PROCESSED / "dataset_enaho_escale_2025.parquet"
    try:
        df.to_parquet(out, index=False)
        LOG.info("Dataset guardado como Parquet: %s", out)
        df.to_csv(PROCESSED / "dataset_enaho_escale_2025.csv", index=False, encoding="utf-8")
    except ImportError:
        out = out.with_suffix(".csv.gz")
        df.to_csv(out, index=False, compression="gzip", encoding="utf-8")
        LOG.warning("pyarrow no disponible; se guardó CSV comprimido: %s", out)
    df.head(0).to_csv(TABLES / "schema_dataset.csv", index=False, encoding="utf-8")
    dictionary = {
        "AÑO": "Año de la encuesta ENAHO.",
        "MES": "Mes de entrevista de la ENAHO.",
        "CONGLOME": "Identificador del conglomerado muestral; unidad de separación entre entrenamiento y prueba.",
        "VIVIENDA": "Identificador de vivienda dentro del conglomerado.",
        "HOGAR": "Identificador de hogar dentro de la vivienda.",
        "UBIGEO": "Código geográfico de ubicación; llave para agregar contexto ESCALE distrital.",
        "CODPERSO": "Identificador de persona dentro del hogar.",
        "P203": "Relación de parentesco con la persona de referencia del hogar.",
        "P207": "Sexo de la persona entrevistada.",
        "P208A": "Edad en años cumplidos.",
        "P300A": "Lengua materna aprendida en la niñez: 1=quechua, 2=aimara, 3=otra nativa, 4=castellano, 6=portugués, 7=otra extranjera, 8=no escucha/habla, 9=lengua de señas, 10=ashaninka, 11=awajún/awarún, 12=shipibo-konibo, 13=shawi, 14=matsigenka, 15=achuar.",
        "P306": "Matrícula en un centro o programa educativo durante el año de referencia (1=sí, 2=no).",
        "P307": "Asistencia actual a un centro o programa educativo (1=sí, 2=no); define población y se excluye de predictores.",
        "P308A": "Nivel en que está matriculada la persona (1=inicial, 2=primaria, 3=secundaria, 4=superior no universitaria, 5=universitaria, 6=maestría/doctorado, 7=básica especial).",
        "P308B": "Año/grado de estudios en que está matriculada la persona.",
        "P308C": "Grado de estudios de primaria; se usa junto con P308B para edad normativa y rezago.",
        "grado_ebr": "Grado válido de primaria (P308C) o secundaria (P308B).",
        "edad_normativa": "Edad normativa derivada: primaria grado+5; secundaria grado+11.",
        "rezago_escolar_anios": "Años de edad por encima de la edad normativa del grado, truncados a cero; faltante si no hay grado válido.",
        "rezago_escolar": "Indicador de rezago: edad mayor que la edad normativa; faltante si no hay grado válido.",
        "P110": "Fuente principal de abastecimiento de agua del hogar; según diccionario, códigos 1 y 2 representan red pública.",
        "P110A1": "Hogar tiene agua potable (1=sí, 2=no).",
        "P111A": "Tipo de servicio higiénico del hogar; códigos 1 y 2 representan conexión a red pública.",
        "agua_potable": "Indicador de agua potable derivado de P110A1; los demás códigos quedan faltantes.",
        "agua_red_publica": "Indicador derivado de P110: fuente de agua en red pública (códigos 1/2).",
        "saneamiento_red_publica": "Indicador derivado de P111A: servicio higiénico conectado a red pública (códigos 1/2).",
        "tic_frecuencia_uso_cluster": "Perfilamiento: P314D (1 diaria, 2 semanal, 3 mensual, 4 cada dos meses o más); 0 si P314A reporta no uso. No se usa en clasificación supervisada.",
        "P314A": "Uso de Internet durante el mes anterior (1=sí, 2=no); define el resultado y se excluye de predictores.",
        "P314D": "Frecuencia de uso de Internet: 1=diaria, 2=semanal, 3=mensual, 4=cada dos meses o más; descriptiva y usada únicamente en clustering, no en clasificación supervisada.",
        "P316B": "Uso de computadora, laptop, tablet o similar en los últimos tres meses (1=sí, 2=no).",
        "tic_habilidades_digitales_n": "Conteo de tareas informáticas afirmativas reportadas (P316C1-P316C10); predictor exploratorio aproximado, no evaluación de competencia certificada.",
        "FACTOR07": "Factor anual de expansión ENAHO basado en proyecciones CPV-2007; ponderador cuando el estimador lo admite.",
        "DOMINIO": "Dominio geográfico de la muestra ENAHO.",
        "ESTRATO": "Estrato muestral ENAHO; base de la derivación del indicador rural.",
        "P1121": "Indicador de alumbrado del hogar mediante electricidad (0=no, 1=sí); no representa acceso a Internet.",
        "P1144": "Indicador de conexión a Internet fija o móvil en el hogar (0=no, 1=sí).",
        "MIEPERHO": "Número de miembros del hogar.",
        "TOTMIEHO": "Total de miembros del hogar según Sumaria.",
        "GASHOG2D": "Gasto total del hogar según Sumaria.",
        "INGHOG2D": "Ingreso total del hogar según Sumaria.",
        "POBREZA": "Condición de pobreza monetaria del hogar según Sumaria.",
        "exclusion_digital": "Indicador: 1=P314A 2 (no usó Internet el mes anterior), 0=P314A 1; población con P307=1 y 6-17 años.",
        "grupo_conglomerado": "CONGLOME conservado como texto; grupo de partición para evitar solapamiento entre train/test.",
        "edad": "Edad en años cumplidos (P208A).",
        "ingreso_hogar_percapita": "INGHOG2D / MIEPERHO.",
        "gasto_hogar_percapita": "GASHOG2D / MIEPERHO.",
        "rural": "Indicador derivado: estratos 6, 7 u 8; verificar codificación anual del marco muestral.",
        "esc_n_escuelas": "Número de COD_MOD distintos en UBIGEO.",
        "esc_n_locales_padlocal": "Número de CODLOCAL distintos observados en Padlocal.dbf por UBIGEO.",
        "esc_alumnos": "Suma de TALUM en padrón agregado por UBIGEO.",
        "esc_matricula_primaria_2025": "Suma distrital del Censo Educativo 2025: C201/3AP (matrícula primaria por grado, sexo y turno), enlazada al padrón por COD_MOD+ANEXO.",
        "esc_matricula_secundaria_2025": "Suma distrital del Censo Educativo 2025: C201/3AS (matrícula secundaria por grado, sexo y turno), enlazada al padrón por COD_MOD+ANEXO.",
        "esc_docentes": "Suma de TDOC en padrón agregado por UBIGEO.",
        "esc_secciones": "Suma de TSEC en padrón agregado por UBIGEO.",
        "esc_ratio_alumno_docente": "Alumnos / docentes agregados del distrito.",
        "esc_prop_gestion_publica": "Proporción de registros cuya etiqueta D_GESTION indica gestión pública; NA/0 si el rótulo no está disponible.",
        "esc_lat_media": "Media de NLAT_IE de instituciones del distrito, cuando está disponible.",
        "esc_lon_media": "Media de NLONG_IE de instituciones del distrito, cuando está disponible.",
        "factor_expansion": "FACTOR07 del módulo Sumaria/educación usado como peso cuando existe.",
    }
    for i, label in enumerate(["hogar", "trabajo", "centro_educativo", "cabina_publica", "casa_otra_persona", "otro_lugar", "acceso_movil"], 1):
        dictionary[f"P314B${i}"] = f"Código de respuesta de lugar de uso de Internet: {i} corresponde a {label}; 0=no marcó el lugar; descriptiva, excluida del modelo por condicionarse al uso."
        dictionary[f"tic_lugar_{i}"] = f"Uso de Internet en {label}, derivado de P314B${i}; descriptivo entre usuarios, excluido del modelo supervisado."
    for i in range(1, 13):
        dictionary[f"P316${i}"] = f"Ítem {i} de propósito de uso de Internet (1=sí, 2=no); descriptivo, excluido del modelo por riesgo de fuga."
        dictionary[f"tic_proposito_{i}"] = f"Propósito de uso de Internet, respuesta afirmativa a P316${i}; descriptivo entre usuarios, excluido del modelo supervisado."
    for i in range(1, 11):
        dictionary[f"P316C{i}"] = f"Tarea informática de habilidades digitales declarada (P316C{i}, 1=sí, 2=no); contribuye al conteo predictor exploratorio."
    dictionary["tic_uso_computadora_ult3m"] = "Uso de computadora/laptop/tablet en los últimos tres meses (derivado de P316B; 1=sí, 0=no)."
    dictionary["P314D"] = "Frecuencia de uso de Internet: 1=diaria, 2=semanal, 3=mensual, 4=cada dos meses o más; solo usuarios."
    pd.DataFrame([{"variable": c, "descripcion": dictionary.get(c, "Variable original ENAHO/ESCALE; ver diccionario anual de la fuente."),
                   "tipo": str(df[c].dtype), "faltantes_pct": round(100*df[c].isna().mean(), 2)} for c in df.columns]).to_csv(TABLES / "diccionario_variables.csv", index=False, encoding="utf-8")
    (TABLES / "resumen_ejecucion.json").write_text(json.dumps({
        "n_filas_analiticas": int(len(df)), "n_columnas": int(len(df.columns)),
        "target_prevalencia": float(df.exclusion_digital.mean()),
        "n_conglomerados": int(df.grupo_conglomerado.nunique()), "semilla": SEED,
        "data_leakage": "GroupShuffleSplit/GroupKFold por CONGLOME; P307/P314A excluidos de predictores; P314B/P316 descriptivos; P314D solo clustering y excluida de clasificación supervisada; imputación, one-hot y escalado dentro de cada fold.",
        "predictores_tic": ["P316B: uso de computadora/tableta", "P316C1-C10: conteo de tareas declaradas"],
        "limitaciones": ["Padlocal.dbf existente sin coordenadas numéricas.",
            "ENAHO no tiene COD_MOD/CODLOCAL: ESCALE se integra por UBIGEO, no por escuela individual.",
            "Archivos disponibles no sustentan indicadores de altitud, equipamiento ni conectividad escolar.",
            "El resultado es no uso de Internet durante el mes previo, no una medida directa de competencia o aprendizaje."]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    save_integration_diagram()
    return df


def feature_columns(df: pd.DataFrame, include_digital_skills: bool = False, include_profile_fields: bool = False,
                    include_geo_id: bool = True) -> tuple[list[str], list[str]]:
    # Se excluyen resultado y variables observadas tras uso de Internet (lugar/frecuencia/propósito).
    categorical = [c for c in ["P203", "P207", "P300A", "P308A", "P110", "P110A1", "P111A", "DOMINIO", "ESTRATO", "UBIGEO"] if c in df]
    if not include_geo_id:
        categorical = [c for c in categorical if c != "UBIGEO"]
    numeric_cols = [c for c in ["edad", "grado_ebr", "rezago_escolar_anios", "P1121", "P1144", "agua_potable", "agua_red_publica", "saneamiento_red_publica", "MIEPERHO", "POBREZA",
        "ingreso_hogar_percapita", "gasto_hogar_percapita", "esc_n_escuelas", "esc_alumnos",
        "esc_n_locales_padlocal",
        "esc_docentes", "esc_secciones", "esc_prop_gestion_publica", "esc_lat_media", "esc_lon_media",
        "esc_ratio_alumno_docente", "esc_matricula_primaria_2025", "esc_matricula_secundaria_2025", "rural"] if c in df]
    if include_digital_skills:
        numeric_cols += [c for c in ["tic_habilidades_digitales_n", "tic_uso_computadora_ult3m"] if c in df]
    if include_profile_fields:
        numeric_cols += [c for c in ["tic_frecuencia_uso_cluster"] if c in df]
    return numeric_cols, categorical


def run_clustering(df: pd.DataFrame) -> None:
    num_cols, cat_cols = feature_columns(df, include_digital_skills=True, include_profile_fields=True, include_geo_id=False)
    cols = num_cols + cat_cols
    if len(df) < 20 or not cols:
        LOG.warning("No hay suficientes filas para clustering.")
        return
    # Ward linkage tiene coste de memoria cuadrático; 4 000 filas permiten comparar
    # ambos algoritmos con recursos de portátil y conservan una muestra reproducible.
    sample = df.sample(n=min(4000, len(df)), random_state=SEED)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat_cols),
    ], remainder="drop")
    X = prep.fit_transform(sample[cols])
    if hasattr(X, "toarray"):
        X = X.toarray()
    # Métricas sobre 20 componentes para retener estructura; PCA 2D se reserva para visualizar.
    pca_model=PCA(n_components=min(20, X.shape[0]-1, X.shape[1]), random_state=SEED)
    cluster_space=pca_model.fit_transform(X)
    pca_2d=PCA(n_components=2, random_state=SEED).fit(X)
    emb=pca_2d.transform(X)
    pd.DataFrame({"componente":np.arange(1,len(pca_model.explained_variance_ratio_)+1),
        "varianza_explicada":pca_model.explained_variance_ratio_,
        "varianza_acumulada":np.cumsum(pca_model.explained_variance_ratio_)}).to_csv(TABLES/"pca_varianza.csv",index=False)
    (TABLES/"configuracion_clustering.json").write_text(json.dumps({
        "muestra_n":int(len(sample)),"semilla":SEED,"variables_numericas":num_cols,"variables_categoricas":cat_cols,
        "identificador_UBIGEO_excluido_como_predictor":True,"componentes_usados_metricas":int(cluster_space.shape[1]),
        "varianza_pca_2d":pca_2d.explained_variance_ratio_.tolist(),
        "nota":"PCA se ajusta tras imputación/estandarización; etiquetas descriptivas y sin validación externa."
    },ensure_ascii=False,indent=2),encoding="utf-8")
    records = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(cluster_space)
        labels = km.labels_
        records.append({"algoritmo":"KMeans", "k":k, "silhouette":silhouette_score(cluster_space, labels, sample_size=min(2000,len(labels)), random_state=SEED),
                        "davies_bouldin":davies_bouldin_score(cluster_space, labels), "calinski_harabasz":calinski_harabasz_score(cluster_space, labels)})
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(cluster_space)
        records.append({"algoritmo":"Agglomerative", "k":k, "silhouette":silhouette_score(cluster_space, ag.labels_, sample_size=min(2000,len(labels)), random_state=SEED),
                        "davies_bouldin":davies_bouldin_score(cluster_space, ag.labels_), "calinski_harabasz":calinski_harabasz_score(cluster_space, ag.labels_)})
    scores = pd.DataFrame(records)
    scores.to_csv(TABLES / "metricas_clustering.csv", index=False)
    winner = scores.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    model = KMeans(n_clusters=int(winner.k), n_init=10, random_state=SEED) if winner.algoritmo == "KMeans" else AgglomerativeClustering(n_clusters=int(winner.k), linkage="ward")
    labels = model.fit_predict(cluster_space)
    sample = sample.copy()
    sample["cluster"] = labels
    profiles=sample.groupby("cluster")[num_cols + ["exclusion_digital"]].mean(numeric_only=True)
    base_rural=float(sample["rural"].mean()) if "rural" in sample else 0.0
    # Use neutral IDs plus explicit evidence; avoid labeling a group as connected
    # based on a tiny P1144 difference when its observed non-use is higher.
    profiles.insert(0,"n",sample.groupby("cluster").size())
    overall_no_use=float(sample["exclusion_digital"].mean())
    def profile_name(row):
        rural_label = "mayor ruralidad" if pd.notna(row.get("rural")) and row["rural"] >= base_rural else "menor ruralidad"
        no_use = row.get("exclusion_digital", np.nan)
        if pd.isna(no_use): use_label = "no uso sin estimar"
        elif no_use >= overall_no_use + .02: use_label = "mayor no uso reciente"
        elif no_use <= overall_no_use - .02: use_label = "menor no uso reciente"
        else: use_label = "no uso cercano al promedio"
        return f"{rural_label.capitalize()} y {use_label} (exploratorio)"
    profiles.insert(0,"nombre_perfil",profiles.apply(profile_name,axis=1))
    profiles.to_csv(TABLES / "perfiles_clusters.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette="tab10", s=18, alpha=.65, ax=ax, legend="full")
    ax.set(title=f"Perfiles de estudiantes: {winner.algoritmo}, k={int(winner.k)}", xlabel="Componente principal 1", ylabel="Componente principal 2")
    fig.tight_layout(); fig.savefig(FIGURES / "clusters_pca.png", dpi=180); plt.close(fig)
    # Silhouette plot for the selected solution, using the same PCA space.
    sil_values = silhouette_samples(cluster_space, labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    lower = 10
    for cluster_id in sorted(np.unique(labels)):
        vals = np.sort(sil_values[labels == cluster_id])
        upper = lower + len(vals)
        ax.fill_betweenx(np.arange(lower, upper), 0, vals, alpha=.75, label=f"Perfil {cluster_id} (n={len(vals)})")
        lower = upper + 10
    ax.axvline(float(np.mean(sil_values)), color="red", linestyle="--", label=f"Media={np.mean(sil_values):.3f}")
    ax.set(title=f"Silueta de la solución seleccionada: {winner.algoritmo}, k={int(winner.k)}", xlabel="Coeficiente de silueta", ylabel="Observaciones ordenadas por perfil")
    ax.legend(loc="best", fontsize=8); fig.tight_layout()
    fig.savefig(FIGURES / "silhouette_clusters.png", dpi=180); plt.close(fig)
    LOG.info("Clustering ganador: %s k=%s", winner.algoritmo, int(winner.k))


def make_preprocessor(num_cols: list[str], cat_cols: list[str], dense: bool = False) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=not dense))]), cat_cols),
    ], remainder="drop", sparse_threshold=0 if dense else 0.3)


def run_supervised(df: pd.DataFrame) -> None:
    num_cols, cat_cols = feature_columns(df, include_digital_skills=True)
    X = df[num_cols + cat_cols].copy()
    y = df["exclusion_digital"].astype(int)
    groups = df["grupo_conglomerado"].astype(str)
    # Test aislado por conglomerado y validación cruzada GroupKFold dentro de train.
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]
    group_overlap=sorted(set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx])))
    (TABLES/"auditoria_split_conglomerado.json").write_text(json.dumps({
        "split":"GroupShuffleSplit","test_size":0.2,"semilla":SEED,"n_train":int(len(train_idx)),"n_test":int(len(test_idx)),
        "conglomerados_train":int(g_train.nunique()),"conglomerados_test":int(groups.iloc[test_idx].nunique()),
        "conglomerados_compartidos_train_test":group_overlap,"separacion_confirmada":len(group_overlap)==0,
        "prevalencia_train":float(y.iloc[train_idx].mean()),"prevalencia_test":float(y.iloc[test_idx].mean())
    },ensure_ascii=False,indent=2),encoding="utf-8")
    class_rows=[]
    for part, idx in [("train",train_idx),("test",test_idx)]:
        yy=y.iloc[idx]; ww=pd.to_numeric(df.iloc[idx].get("factor_expansion",pd.Series(np.nan,index=idx)),errors="coerce")
        ww=ww.fillna(ww.median() if ww.notna().any() else 1).clip(lower=0)
        for label in (0,1):
            mask=yy.eq(label)
            class_rows.append({"particion":part,"clase":label,"n":int(mask.sum()),"porcentaje":float(100*mask.mean()),"peso_factor07_sumado":float(ww[mask].sum())})
    pd.DataFrame(class_rows).to_csv(TABLES/"distribucion_clases.csv",index=False,encoding="utf-8")
    n_splits = min(3, g_train.nunique())
    if n_splits < 2:
        LOG.warning("Menos de dos conglomerados; se omite clasificación."); return
    cv = GroupKFold(n_splits=n_splits)
    models = {
        "LogisticRegression_baseline": (__import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=800, class_weight="balanced", random_state=SEED), [{}]),
        "RandomForest": (RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
                         [{"n_estimators":150,"min_samples_leaf":3},{"n_estimators":250,"min_samples_leaf":6}]),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED),
                       [{"n_estimators":150,"min_samples_leaf":3},{"n_estimators":250,"min_samples_leaf":6}]),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=100, learning_rate=.08, l2_regularization=1.0, class_weight="balanced", random_state=SEED),
                                 [{"max_iter":100,"learning_rate":.08},{"max_iter":150,"learning_rate":.05}]),
    }
    results=[]; fitted={}; selected_params={}; selected_cv_rows=[]
    sw = pd.to_numeric(df.iloc[train_idx].get("factor_expansion", pd.Series(index=train_idx, dtype=float)), errors="coerce")
    sample_weight = sw.fillna(sw.median() if sw.notna().any() else 1).clip(lower=0).to_numpy()
    for name, (estimator, candidates) in models.items():
        try:
            candidate_results=[]
            for params in candidates:
                tuned_estimator=clone(estimator).set_params(**params)
                candidate_pipe=Pipeline([("preprocess", make_preprocessor(num_cols, cat_cols, dense=(name=="HistGradientBoosting"))), ("model", tuned_estimator)])
                fold_rows=[]
                for fold_id, (fold_train, fold_valid) in enumerate(cv.split(X_train,y_train,groups=g_train), start=1):
                    fold_model=clone(candidate_pipe)
                    fold_model.fit(X_train.iloc[fold_train],y_train.iloc[fold_train],model__sample_weight=sample_weight[fold_train])
                    fold_probability=fold_model.predict_proba(X_train.iloc[fold_valid])[:,1]
                    fold_y=y_train.iloc[fold_valid]; fold_w=sample_weight[fold_valid]
                    fold_pred=(fold_probability>=.5).astype(int)
                    fold_pr,fold_rc,_=precision_recall_curve(fold_y,fold_probability,sample_weight=fold_w)
                    fold_rows.append({"modelo":name,"fold":fold_id,
                        "roc_auc":roc_auc_score(fold_y,fold_probability,sample_weight=fold_w) if fold_y.nunique()>1 else np.nan,
                        "pr_auc":auc(fold_rc,fold_pr),"f1_macro":f1_score(fold_y,fold_pred,average="macro",sample_weight=fold_w),
                        "n_train":int(len(fold_train)),"n_valid":int(len(fold_valid)),
                        "grupos_train":int(g_train.iloc[fold_train].nunique()),"grupos_valid":int(g_train.iloc[fold_valid].nunique()),
                        "prevalencia_valid_ponderada":float(np.average(fold_y,weights=fold_w))})
                fold_frame=pd.DataFrame(fold_rows)
                candidate_results.append((fold_frame,params,candidate_pipe))
            cv_metrics,best_params,pipe=max(candidate_results,key=lambda item:item[0].roc_auc.mean())
            selected_cv_rows.extend(cv_metrics.to_dict("records"))
            pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
            proba = pipe.predict_proba(X_test)[:, 1]
            weight_test_raw = pd.to_numeric(df.iloc[test_idx].get("factor_expansion", pd.Series(index=test_idx, dtype=float)), errors="coerce")
            weight_test = weight_test_raw.fillna(weight_test_raw.median() if weight_test_raw.notna().any() else 1).clip(lower=0).to_numpy()
            pred = (proba >= .5).astype(int)
            pr, rc, _ = precision_recall_curve(y_test, proba, sample_weight=weight_test)
            results.append({"modelo":name, "cv_roc_auc_mean":cv_metrics.roc_auc.mean(), "cv_roc_auc_sd":cv_metrics.roc_auc.std(),
                "cv_pr_auc_mean":cv_metrics.pr_auc.mean(),"cv_pr_auc_sd":cv_metrics.pr_auc.std(),
                "cv_f1_macro_mean":cv_metrics.f1_macro.mean(),"cv_f1_macro_sd":cv_metrics.f1_macro.std(),
                "test_roc_auc":roc_auc_score(y_test, proba, sample_weight=weight_test) if y_test.nunique()>1 else np.nan,
                "test_pr_auc":auc(rc, pr), "test_f1_macro":f1_score(y_test,pred,average="macro",sample_weight=weight_test),
                "test_precision":precision_score(y_test,pred,zero_division=0,sample_weight=weight_test), "test_recall":recall_score(y_test,pred,zero_division=0,sample_weight=weight_test),
                "test_brier":brier_score_loss(y_test,proba,sample_weight=weight_test)})
            fitted[name]=pipe
            selected_params[name]=best_params
            results[-1]["hiperparametros_seleccionados"]=json.dumps(best_params,ensure_ascii=False,sort_keys=True)
            LOG.info("Modelo %s evaluado.", name)
        except Exception as exc:
            LOG.exception("Modelo %s no se pudo ajustar: %s", name, exc)
    if not results:
        return
    scores=pd.DataFrame(results).sort_values("cv_roc_auc_mean",ascending=False)
    scores.to_csv(TABLES/"comparacion_modelos.csv",index=False)
    pd.DataFrame(selected_cv_rows).to_csv(TABLES/"cv_groupkfold_modelos.csv",index=False,encoding="utf-8")
    best_name=str(scores.iloc[0].modelo); best=fitted[best_name]
    pred=best.predict(X_test); proba=best.predict_proba(X_test)[:,1]
    pd.DataFrame(confusion_matrix(y_test,pred),index=["real_0","real_1"],columns=["pred_0","pred_1"]).to_csv(TABLES/"matriz_confusion.csv")
    test_weight_raw = pd.to_numeric(df.iloc[test_idx].get("factor_expansion", pd.Series(index=test_idx, dtype=float)), errors="coerce")
    weight_test = test_weight_raw.fillna(test_weight_raw.median() if test_weight_raw.notna().any() else 1).clip(lower=0).to_numpy()
    pd.DataFrame(confusion_matrix(y_test,pred,sample_weight=weight_test),index=["real_0_ponderado","real_1_ponderado"],columns=["pred_0_ponderado","pred_1_ponderado"]).to_csv(TABLES/"matriz_confusion_ponderada.csv")
    (TABLES/"reporte_clasificacion.txt").write_text(classification_report(y_test,pred,zero_division=0),encoding="utf-8")
    # Curvas ROC y PR + calibración. Datos de prueba permanecen agrupados fuera del entrenamiento.
    fig, axs=plt.subplots(1,3,figsize=(15,4.5))
    fpr,tpr,_=roc_curve(y_test,proba,sample_weight=weight_test); axs[0].plot(fpr,tpr,label=f"AUC ponderado={roc_auc_score(y_test,proba,sample_weight=weight_test):.3f}"); axs[0].plot([0,1],[0,1],"--",color="grey"); axs[0].set(title="ROC",xlabel="FPR",ylabel="TPR"); axs[0].legend()
    pr,rc,_=precision_recall_curve(y_test,proba,sample_weight=weight_test); axs[1].plot(rc,pr); axs[1].set(title="Precision-Recall ponderada",xlabel="Recall",ylabel="Precision")
    bin_edges=np.linspace(0,1,11)
    bin_ids=np.clip(np.digitize(proba,bin_edges[1:-1],right=False),0,9)
    prob_true=[]; prob_pred=[]
    for bin_id in range(10):
        selected=bin_ids==bin_id
        if selected.any():
            local_weight=weight_test[selected]
            prob_pred.append(np.average(proba[selected],weights=local_weight))
            prob_true.append(np.average(y_test.to_numpy()[selected],weights=local_weight))
    axs[2].plot(prob_pred, prob_true, marker="s", label="Clasificador ponderado")
    axs[2].plot([0,1],[0,1],":",color="black",label="Calibración perfecta")
    axs[2].set(title="Calibración ponderada",xlabel="Probabilidad media predicha",ylabel="Fracción positiva"); axs[2].legend()
    fig.tight_layout(); fig.savefig(FIGURES/"evaluacion_modelos.png",dpi=180); plt.close(fig)
    import joblib
    joblib.dump(best,MODELS/"modelo_ganador.joblib")
    (MODELS/"metadatos_modelo.json").write_text(json.dumps({"modelo":best_name,"hiperparametros":selected_params[best_name],"busqueda":"candidatos predefinidos evaluados con GroupKFold ponderado; el test por CONGLOME permanece separado","semilla":SEED,"features_numericas":num_cols,"features_categoricas":cat_cols,"n_train":len(train_idx),"n_test":len(test_idx),"n_grupos_train":int(g_train.nunique()),"target":"exclusion_digital"},ensure_ascii=False,indent=2),encoding="utf-8")
    explain_shap(best,X_train,X_test,y_test,best_name)


def explain_shap(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """SHAP global por variable fuente y extremos del conjunto de prueba completo."""
    try:
        import shap
        prep=pipe.named_steps["preprocess"]; estimator=pipe.named_steps["model"]
        bg=X_train.sample(n=min(100,len(X_train)),random_state=SEED)
        # Selecciona extremos correctos en TODO el test; los añade a la muestra explicada.
        test_probability=pipe.predict_proba(X_test)[:,1]
        test_prediction=(test_probability>=.5).astype(int)
        y_test_values=y_test.to_numpy()
        high_candidates=np.flatnonzero((test_prediction==1)&(y_test_values==1))
        low_candidates=np.flatnonzero((test_prediction==0)&(y_test_values==0))
        high_id=X_test.index[high_candidates[np.argmax(test_probability[high_candidates])]] if len(high_candidates) else None
        low_id=X_test.index[low_candidates[np.argmin(test_probability[low_candidates])]] if len(low_candidates) else None
        explain_ids=list(X_test.sample(n=min(250,len(X_test)),random_state=SEED).index)
        explain_ids=list(dict.fromkeys(explain_ids+[x for x in [high_id,low_id] if x is not None]))
        ex=X_test.loc[explain_ids]
        bgx=prep.transform(bg); exx=prep.transform(ex)
        if hasattr(bgx,"toarray"): bgx=bgx.toarray()
        if hasattr(exx,"toarray"): exx=exx.toarray()
        names=prep.get_feature_names_out()
        explainer=shap.TreeExplainer(estimator) if hasattr(estimator,"estimators_") or hasattr(estimator,"tree_") else shap.Explainer(estimator.predict_proba,bgx)
        vals=explainer(exx)
        if hasattr(vals,"feature_names") and vals.feature_names is None:
            vals.feature_names=list(names)
        sv=vals[...,1] if len(vals.shape)>2 else vals
        # Reagrupa los dummies y marcadores de faltante por variable original.
        transformed_names=list(names); source_cols=list(X_train.columns); grouped_indices={c:[] for c in source_cols}
        for idx,name in enumerate(transformed_names):
            raw=name.replace("num__","").replace("cat__","")
            if raw.startswith("missingindicator_"): raw=raw[len("missingindicator_"):]
            matches=[c for c in source_cols if raw==c or raw.startswith(c+"_")]
            if matches: grouped_indices[max(matches,key=len)].append(idx)
        grouped_cols=[c for c in source_cols if grouped_indices[c]]
        grouped_values=np.column_stack([np.asarray(sv.values)[:,grouped_indices[c]].sum(axis=1) for c in grouped_cols])
        grouped_exp=shap.Explanation(values=grouped_values,feature_names=grouped_cols)
        shap.plots.beeswarm(grouped_exp,max_display=min(11,len(grouped_cols)),show=False); plt.tight_layout(); plt.savefig(FIGURES/"shap_beeswarm.png",dpi=180,bbox_inches="tight"); plt.close()
        importance=np.abs(grouped_values).mean(axis=0)
        importance_table=pd.DataFrame({"variable":grouped_cols,"mean_abs_shap":importance,
            "metodo":"contribuciones SHAP sumadas por variable fuente, luego valor absoluto y promedio"}).sort_values("mean_abs_shap",ascending=False).head(10)
        importance_table.to_csv(TABLES/"shap_importancia_top10.csv",index=False,encoding="utf-8")
        feature_names=list(names)
        base_idx=next((i for i,n in enumerate(feature_names) if n.endswith("P1144")), None)
        descriptions=[]
        if base_idx is not None and "P1144" in ex:
            raw_x=pd.to_numeric(ex["P1144"],errors="coerce").fillna(0).to_numpy()
            raw_language=pd.to_numeric(ex.get("P300A",pd.Series(np.nan,index=ex.index)),errors="coerce")
            lang_color=np.select([raw_language.eq(1),raw_language.eq(2)],[1,2],default=0)
            fig,ax=plt.subplots(figsize=(7,5)); sc=ax.scatter(raw_x,np.asarray(sv.values)[:,base_idx],c=lang_color,cmap="viridis",vmin=0,vmax=2,s=28,alpha=.7)
            cb=fig.colorbar(sc,ax=ax,ticks=[0,1,2]); cb.ax.set_yticklabels(["Otras / faltante","Quechua","Aimara"])
            ax.set(xticks=[0,1],xlabel="Internet en el hogar (P1144: 0=no, 1=sí)",ylabel="Contribución SHAP a la predicción",title="P1144 y lengua materna (categorías ENAHO)")
            fig.tight_layout(); fig.savefig(FIGURES/"shap_dependence_1.png",dpi=180,bbox_inches="tight"); plt.close(fig)
            descriptions.append({"grafico":1,"eje_x":"P1144 original (0=no Internet en hogar; 1=sí)","color":"P300A original: Quechua, Aimara, otras lenguas/faltante","n":int(len(ex)),"unidad_y":"valor SHAP del predictor P1144"})
            if "esc_ratio_alumno_docente" in ex:
                ratio=pd.to_numeric(ex["esc_ratio_alumno_docente"],errors="coerce")
                ratio=ratio.fillna(pd.to_numeric(X_train["esc_ratio_alumno_docente"],errors="coerce").median()).to_numpy()
                fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(ratio,np.asarray(sv.values)[:,base_idx],c=raw_x,cmap="coolwarm",vmin=0,vmax=1,s=28,alpha=.7)
                cb=fig.colorbar(ax.collections[0],ax=ax,ticks=[0,1]); cb.ax.set_yticklabels(["Sin Internet hogar","Con Internet hogar"])
                ax.set(xlabel="Ratio alumnos/docente distrital (Padron)",ylabel="Contribución SHAP de P1144",title="Interacción exploratoria: Internet del hogar y contexto escolar")
                fig.tight_layout(); fig.savefig(FIGURES/"shap_dependence_2.png",dpi=180,bbox_inches="tight"); plt.close(fig)
                descriptions.append({"grafico":2,"eje_x":"ratio distrital alumnos/docente en unidades originales","color":"P1144 original: internet en hogar","n":int(len(ex)),"unidad_y":"valor SHAP del predictor P1144"})
        (TABLES/"shap_dependencias.json").write_text(json.dumps(descriptions,ensure_ascii=False,indent=2),encoding="utf-8")
        # Casos extremos correctamente clasificados en la totalidad del test.
        pred=pipe.predict(ex); prob=pipe.predict_proba(ex)[:,1]; actual=y_test.loc[ex.index].to_numpy()
        local_cases=[]
        for case_name,target_id in [("alto",high_id),("bajo",low_id)]:
            if target_id is None: continue
            pos=list(ex.index).index(target_id); shap.plots.waterfall(sv[int(pos)],max_display=10,show=False)
            plt.tight_layout(); plt.savefig(FIGURES/f"shap_waterfall_{case_name}.png",dpi=180,bbox_inches="tight"); plt.close()
            contributions=np.asarray(sv.values)[int(pos)]; top_idx=np.argsort(np.abs(contributions))[::-1][:6]
            local_cases.append({"caso":case_name,"seleccion":"extremo correcto entre todos los casos del test agrupado",
                "real":int(actual[pos]),"predicho":int(pred[pos]),"probabilidad":float(prob[pos]),"indice_fila":str(target_id),
                "contribuciones_top":[{"variable":str(names[i]),"shap":float(contributions[i]),"valor_transformado":float(exx[pos,i])} for i in top_idx]})
        (TABLES/"shap_casos_locales.json").write_text(json.dumps(local_cases,ensure_ascii=False,indent=2),encoding="utf-8")
        LOG.info("Explicabilidad SHAP generada para %s.",model_name)
        (REPORTS/"shap_estado.txt").unlink(missing_ok=True)
    except Exception as exc:
        LOG.exception("SHAP no se pudo generar: %s",exc)
        (REPORTS/"shap_estado.txt").write_text(f"No fue posible calcular SHAP: {exc}\n",encoding="utf-8")


def main() -> None:
    df=build_dataset()
    if df.empty:
        raise RuntimeError("El cruce no produjo observaciones escolares con asistencia e información de Internet.")
    LOG.info("Dataset analítico: %s filas; %s columnas; target positivo=%.3f",f"{len(df):,}",len(df.columns),df.exclusion_digital.mean())
    df.sample(n=min(100000,len(df)),random_state=SEED).to_csv(TABLES/"muestra_auditoria.csv",index=False,encoding="utf-8")
    run_clustering(df)
    run_supervised(df)
    summary={"n_filas_analiticas":int(len(df)),"n_columnas":int(len(df.columns)),
        "target_prevalencia":float(df.exclusion_digital.mean()),"n_conglomerados":int(df.grupo_conglomerado.nunique()),"semilla":SEED,
        "data_leakage":"GroupShuffleSplit/GroupKFold por CONGLOME; P307/P314A excluidos de predictores; P314B/P316 descriptivos; P314D solo clustering y excluida de clasificación supervisada; imputación, one-hot y escalado dentro de cada fold.",
        "predictores_tic":["P316B: uso de computadora/tableta","P316C1-C10: conteo de tareas digitales autodeclaradas"],
        "variables_cluster":["ingreso/gasto per cápita","ratio alumnos/docente distrital","frecuencia TIC P314D con 0=no uso","rezago escolar en años","acceso a agua, saneamiento y electricidad"],
        "limitaciones":["Padlocal.dbf existente sin coordenadas numéricas.","ENAHO no tiene COD_MOD/CODLOCAL: ESCALE se agrega por UBIGEO y no identifica la escuela individual.",
            "Los archivos disponibles no sustentan indicadores escolares de altitud, equipamiento informático ni conectividad.",
            "La matrícula ESCALE se vincula entre archivos escolares por COD_MOD+ANEXO; el resultado final es contexto distrital.",
            "El resultado es no uso de Internet durante el mes previo, no una medida directa de competencia o aprendizaje."]}
    (TABLES/"resumen_ejecucion.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    from src.report import generate_report
    LOG.info("Informe técnico generado: %s", generate_report())
    LOG.info("Pipeline finalizado. Consulte outputs/ y data/processed/.")


if __name__ == "__main__":
    main()
