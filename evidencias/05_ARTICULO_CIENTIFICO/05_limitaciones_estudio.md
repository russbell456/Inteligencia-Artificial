# Limitaciones verificadas

- El enlace individual escolar no puede hacerse: ENAHO no incluye COD_MOD/CODLOCAL. ESCALE es contexto distrital.
- Padlocal disponible carece de coordenadas numéricas/altitud; los archivos incluidos no sustentan altitud, equipamiento informático o conectividad escolar.
- `exclusion_digital` mide no uso declarado de Internet en el mes anterior, no competencia ni desempeño escolar directo.
- Los perfiles son exploratorios y dependen de muestra/variables; nombres no deben presentarse como categorías validadas.
- PR-AUC del pipeline es integral trapezoidal de la curva precisión-recall; Average Precision es una definición distinta.
- ROC/PR/métricas usan factor de expansión donde se documenta; conteos de matriz de confusión son no ponderados. No se estiman intervalos de confianza del diseño complejo.
- La definición de salida/background SHAP no queda completamente especificada para sostener interpretaciones probabilísticas.
- No se encontró evidencia exportada de una curva de calibración fuente; esta auditoría la derivó del modelo/split guardados sin reentrenar.
- La guía del docente no se localizó en el directorio del proyecto durante esta auditoría; las referencias rectoras disponibles son las instrucciones ya citadas en la conversación y matriz local de cumplimiento. Confirmar PDF fuente antes de redacción final.
