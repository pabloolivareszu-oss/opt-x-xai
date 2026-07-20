# Estado del proyecto — Framework XAI como control adaptativo en metaheurísticas

Tesis: el XAI (SHAP / LIME / ACME / iBreakdown) actúa como **oráculo** que, ante
estancamiento, estima la importancia de los parámetros del algoritmo y genera un
*counterfactual* determinista que los reajusta. Dos escenarios: una **campaña
principal** (benchmark sintético CEC2022) y un **experimento aplicado**
(problemas de ingeniería del mundo real, CEC2020 RW-CMOP).

---

## 1. Escenario experimental

### 1.1 Campaña principal (CEC2022)

| Componente | Valor |
|---|---|
| Benchmark | IEEE CEC 2022 (single-objective bound-constrained) |
| Algoritmos | 8 (PSO, BA, GWO, L-SHADE, jSO, GSK, OPA, SBOA) |
| Variantes (5) | Standard, SHAP-CF, LIME-CF, ACME-CF, IBREAKDOWN-CF |
| Funciones | 12 (F1–F12) |
| Dimensiones | D=10, D=20 |
| (nota) | el CSV parcial en disco trae 10 de las 12 (faltan F2, F5) |
| Runs | 30 |
| MaxFEs | 200.000 (D=10) · 1.000.000 (D=20) |
| Target error | 1e-8 · semillas pareadas (CRN) |
| **Total runs** | **28.800** |

### 1.2 Experimento aplicado (RW-CMOP)

| Componente | Valor |
|---|---|
| Benchmark | CEC 2020 Real-World Constrained (Kumar et al.) |
| Algoritmos | 2 (GSK, SBOA) |
| Variantes (5) | Standard, SHAP-CF, LIME-CF, ACME-CF, IBREAKDOWN-CF |
| Problemas (10) | RW02, RW08, RW14, RW16, RW17, RW23, RW26, RW40, RW41, RW45 |
| Dimensiones | nativas por problema (D ∈ {2,3,4,5,7}) |
| Runs | 50 |
| MaxFEs | 200.000 (fijo, alineado con D=10 principal) |
| Semillas | pareadas (CRN) |
| **Total runs** | **5.000** |

Manejo de restricciones: regla de factibilidad de Deb. Escalarización
penalizada `f + RHO·CV` (RHO=1000) para el motor; el f/cv reportado se
re-evalúa sobre `best_x` (mejor vector global, no poblacional — verificado).

---

## 2. Datos disponibles actualmente

### 2.1 Campaña principal
- `Table_Performance.csv` — **subconjunto** en disco: 6 algoritmos
  (BA, GSK, L-SHADE, PSO, SBOA, jSO) de los 8, 5 variantes, 10 de las 12
  funciones (faltan F2 y F5), D=10/20, 2.700 filas. Incluye Final_Fit,
  Final_Err, **Best_X_JSON**, FEs por categoría
  (Optimizer/Explanation/DirectionProbe/CFValidation/Oracle), trayectoria
  (TrajectorySnapshots), diversidad final, activaciones/rescates/intervenciones.
- `Table_Statistics_Pairwise_Standard_vs_Hybrid.csv` — comparaciones pareadas.
- `tabla_error_stats.tex` — **análisis del top (BESTX)**: tabla en español de
  estadísticos de error final `|f_best − f*|` (30 runs) **solo para las celdas
  con mejora XAI significativa** (Wilcoxon + Holm), con el mejor valor por
  columna en negrita y "Estándar" como línea base. Es la tabla que destaca
  dónde el oráculo sí gana (p.ej. BA F3 D=10 → ACME mejor en Best/Mediana/
  Media/Worst; BA F6 D=20 → ACME; BA F10 D=20 → ACME).
- `stats_table_full.csv` — estadísticos completos por celda (algoritmo, dim,
  función, variante): Best/Worst/Median/Mean/Std/Q1/Q3, 91 filas. Base del
  análisis del top.

### 2.2 Experimento aplicado (RW-CMOP)
Generados por `run_rwcmop_real.py` (motor real). De la corrida previa (2
problemas, 500 runs) y la nueva (10 problemas, 5.000 runs):
- `per_run.csv` — f, cv, feasible, activaciones, intervenciones, rescates,
  fallidas, diversidad, tiempo, **best_x_json** por corrida.
- `convergence.csv` — best-so-far (nfe, best_fit) submuestreado por corrida.
- `interventions.csv` — un registro por activación (trigger, applied, useful,
  realized_gain, status).
- `trajectory.csv` — traza por generación: FEs por categoría, diversidad,
  estancamiento, y **parámetros accionables** (GSK: KF/KR; SBOA: hunt/escape/
  explore/local weights).
- `best_solutions.csv` — mejor vector (best_x_json) por celda según Deb.
- `summary.csv` — best/median/mean/std/worst, CV media, % factible,
  activaciones y rescates medios.
- `meta.json` — configuración y `oracle_config` real.

### 2.3 Análisis y figuras ya producidos
- `analyze_rwcmop.py` → `stats_table.csv/.tex`, `oracle_dynamics.csv`,
  `tests_wilcoxon_holm.csv`, `convergence_mean.csv`, `RESUMEN_comportamiento.md`.
- `plotter_rwcmop.py` → convergencia (X normalizado, sin bandas, estilo
  campaña principal), boxplots, dinámica del oráculo (300 dpi PNG).

---

## 3. Qué falta para un análisis COMPLETO

### 3.1 Campaña principal
- [ ] Confirmar si el `Table_Performance.csv` completo (8 algoritmos, 12
  funciones, 28.800 runs) está disponible o si solo existe el subconjunto de
  2.700 filas (6 algoritmos, 10 funciones). El análisis debe declarar el
  alcance real.
- [ ] Estadísticos por (algoritmo, variante, función, D): best/median/mean/std/
  worst del error, tasa de éxito (TargetReached), FE al primer target.
- [ ] Tests: Wilcoxon pareado Standard-vs-variante por celda; Friedman + Holm
  por algoritmo a través de funciones; ranking global.
- [ ] Curvas de convergencia agregadas (ya hay plotter; falta el CSV de
  convergencia de la campaña principal, no solo el de RW-CMOP).
- [ ] Dinámica del oráculo: relación activaciones→rescates por algoritmo;
  reparto de presupuesto (ExplanationShareOfConsumedFEs).
- [ ] Extender el **análisis del top (BESTX)** ya iniciado (`tabla_error_stats`)
  al conjunto completo: identificar todas las celdas con mejora significativa,
  qué explainer gana en cada una, y consolidar el ranking de "mejores
  configuraciones" por algoritmo. Es la evidencia directa de C1 y C2'.

### 3.2 Experimento aplicado
- [ ] Re-correr `analyze_rwcmop.py` sobre los 5.000 runs nuevos (el análisis
  actual es de los 500 previos). Genera tabla, tests, dinámica, convergencia.
- [ ] Tests por problema (10) con marca de "muestras idénticas" donde el
  oráculo no se activa.
- [ ] Tabla de mejores soluciones decodificadas (best_x) por problema.

### 3.3 Lectura de parámetros y sus límites (análisis aparte)
- [ ] Cruzar `trajectory.csv` con los **límites accionables** de cada
  algoritmo para ver dónde el oráculo empuja los parámetros:
  - GSK: KF ∈ [0.1, 1.0], KR ∈ [0.1, 0.9] (de `actionable_bounds`).
  - SBOA: hunt/escape/explore/local weights (rangos en su engine).
- [ ] Verificar si las intervenciones llevan los parámetros a los bordes
  (saturación) o a zonas interiores, y si eso correlaciona con rescate útil.
- [ ] `intervention_strength=0.1` y `cf_configurations_per_activation=1`:
  evaluar si el paso de intervención es suficiente para mover la búsqueda.

---

## 4. Configuración del oráculo (límites operativos)

Valores reales de `DEFAULT_ORACLE_CONFIG` (para interpretar la dinámica):

| Parámetro | Valor | Rol |
|---|---|---|
| trigger_window_fraction | 0.02 | ventana de detección de estancamiento |
| history_window_fraction | 0.02 | ventana de historia para el explainer |
| post_window_fraction | 0.01 | ventana de evaluación post-intervención |
| cooldown_fraction | 0.01 | enfriamiento entre activaciones |
| probe_population_cap | 20 | tope de población para sondeo |
| shap_queries / lime_queries | 16 / 12 | consultas del explainer |
| acme_grid_points_per_feature | 3 | resolución ACME |
| ibreakdown_max_features / pair_checks | 6 / 4 | profundidad iBreakdown |
| cf_configurations_per_activation | 1 | counterfactuals por activación |
| intervention_strength | 0.1 | magnitud del reajuste |
| meaningful_improvement_rel_threshold | 1e-8 | umbral de mejora significativa |

---

## 5. Conjeturas (a contrastar con los datos completos)

Marco de hipótesis para el análisis. Cada una se confirma/refuta con los
estadísticos y tests, sin asumir el resultado.

- **C1 — Selectividad.** El beneficio del XAI no es universal: aparece en
  algunos pares (algoritmo, función/problema) y no en otros.
- **C2 — No hay explainer universal.** Ningún explainer (SHAP/LIME/ACME/
  iBreakdown) domina en todos los casos; el mejor depende del contexto.
- **C2' — Descomposición vs perturbación.** Los explainers de descomposición
  (ACME, iBreakdown) y los perturbativos (SHAP, LIME) tendrían perfiles de
  mejora distintos según el algoritmo.
- **C3 — Receptividad por algoritmo.** Algunos algoritmos (p.ej. GSK) serían
  más receptivos a la intervención que otros (p.ej. GWO/OPA sin señal).
- **C4 — Activación ≠ rescate.** Una alta tasa de activación no implica
  rescate: el oráculo puede intervenir mucho y mejorar poco (medible con
  activaciones vs rescates y realized_gain).
- **C5 — Dependencia del paisaje.** El XAI rescata en paisajes multimodales
  engañosos (CEC2022) pero su efecto se atenúa en paisajes de descenso suave o
  de factibilidad dura (observado en el aplicado).
- **C6 — Límite honesto.** El XAI puede ser neutro o contraproducente; el
  framework delimita su propio dominio de aplicabilidad en lugar de prometer
  mejora universal.

---

## 6. Orden sugerido de trabajo (por partes)

1. Confirmar alcance real de la campaña principal (subconjunto vs completo).
2. Análisis estadístico de la campaña principal (tabla + tests + ranking).
3. Re-análisis del aplicado sobre los 5.000 runs.
4. Lectura de parámetros vs límites accionables (trajectory.csv).
5. Cierre: contraste de conjeturas C1–C6 con la evidencia.

(Los plotters ya están listos para ambos escenarios.)
