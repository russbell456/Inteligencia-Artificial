# Instrucciones de reproducibilidad

Este paquete documenta, no reemplaza, el proyecto original. Para repetir únicamente las figuras derivadas de curvas en el entorno del proyecto, desde la raíz `data_limpia` ejecute:

```powershell
.\.venv\Scripts\python.exe .\EVIDENCIAS_FINALES_IA_2025\06_REPRODUCIBILIDAD\generar_evidencias.py
```

El script toma el dataset y `modelo_ganador.joblib` guardados en `entregables/`, calcula curvas agregadas y figuras; no ajusta modelos ni exporta microdatos. El resto de evidencia tabular fue copiada de resultados existentes.

El proyecto completo se ejecuta por separado desde `entregables/` siguiendo `README.md` y `requirements-lock.txt`. **No ejecutar `src.pipeline` para esta auditoría**, porque volvería a escribir los artefactos originales.
