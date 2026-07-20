# X-Opt Oracle

**Marco de trabajo supervisor XAI-contrafactual para el ajuste dinámico de parámetros en algoritmos de optimización global.**

Pablo Olivares, Rodrigo Olivares — Escuela de Ingeniería Informática, Universidad de Valparaíso, Chile.

---

## Resumen

Las metaheurísticas poblacionales suelen estancarse en óptimos locales sin mecanismos que expliquen por qué ocurre ni que orienten su recuperación. **X-Opt Oracle** es una capa supervisora desacoplada que monitorea al algoritmo anfitrión y, ante una señal de estancamiento, ejecuta una secuencia diagnóstica-prescriptiva:

1. Construye un **oráculo empírico pagado** $\mathcal{M}(\Theta)$ que re-simula horizontes cortos de búsqueda, cargando cada evaluación al mismo presupuesto global.
2. Estima la **importancia de los parámetros accionables** mediante un explicador XAI intercambiable (SHAP, LIME, ACME o iBreakdown).
3. Sintetiza una **reconfiguración contrafactual determinista** y la aplica como override persistente.
4. **Audita** el efecto posterior (rescate, neutralidad o interferencia).

Todo el proceso opera bajo un **ledger estricto de evaluaciones** con pre-reserva presupuestaria: ninguna mejora proviene de evaluaciones ocultas.

El hallazgo central es que la asistencia XAI-CF **no produce mejoras universales, sino beneficios selectivos** determinados por la *plasticidad paramétrica* del anfitrión: SBOA, GSK y BA son receptivos; los mecanismos auto-adaptativos de L-SHADE y jSO amortiguan la intervención; PSO permanece neutro; GWO y OPA delimitan la zona de interferencia. ACME-CF es el explicador con mejor rango global entre los anfitriones receptivos. En el traslado aplicado (planificación de un hub de carga de vehículos eléctricos), GSK–ACME-CF supera estadísticamente a Standard en los diez escenarios evaluados, con mejoras de magnitud moderada.

---

## Contribuciones

- La explicabilidad deja de ser inspección post-hoc y se convierte en **componente operativo de control de lazo cerrado**.
- Arquitectura **agnóstica al explicador y al anfitrión**: 8 metaheurísticas × 4 explicadores bajo la misma contabilidad.
- **Auditabilidad formal**: ledger con invariante de suma exacta (Proposición 1) y cota de actividad del oráculo (Proposición 2).
- Caracterización empírica de la **frontera de aplicabilidad**: cuándo la intervención rescata, cuándo es neutra y cuándo interfiere.

---

## Matemáticas del oráculo

### Problema y estancamiento

Optimización global continua sobre $\Omega = \prod_{j=1}^{D}[L_j, U_j]$:

$$\min_{x \in \Omega} f(x)$$

Sea $\tau^\star(t)$ la última evaluación con mejora relativa del incumbente superior a $\delta = 10^{-8}$ (considerando solo evaluaciones del optimizador). El disparador de estancamiento es:

$$
I_{stag}(t) =
\begin{cases}
1 & \text{si } t - \tau^\star(t) \ge W_{trig} \ \wedge \ t \ge t_{cd} \\
0 & \text{en otro caso}
\end{cases}
\qquad W_{trig} = \lceil 0{,}02 \cdot MaxFEs \rceil
$$

con cooldown $W_{cd} = \lceil 0{,}01 \cdot MaxFEs \rceil$ tras intervenir ($W_{cd}/3$ tras un intento bloqueado). La diversidad poblacional se registra como telemetría:

$$\sigma^2(P_t) = \frac{1}{N \cdot D} \sum_{i=1}^{N} \sum_{j=1}^{D} (x_{i,j} - \bar{x}_j)^2$$

### Ledger estricto y gate de disponibilidad

Toda evaluación real se imputa a una de cuatro categorías:

$$FE^{tot}(t) = FE^{opt}(t) + FE^{exp}(t) + FE^{dir}(t) + FE^{cf}(t) \le MaxFEs$$

(en la implementación evaluada $FE^{dir} = FE^{cf} = 0$; la dirección se infiere sin coste desde el historial). Antes de activar el explicador $v$ sobre $m$ parámetros, el gate exige la pre-reserva:

$$MaxFEs - FE^{tot}(t) \ \ge\ R_v(m) = Q_v(m) \cdot P$$

**Proposición 1 (seguridad presupuestaria).** Bajo el ledger con pre-reserva, toda corrida satisface $FE^{tot} \le MaxFEs$ y la suma de categorías coincide exactamente con $FE^{tot}$ (verificado con aserción al cierre de cada corrida).

### Oráculo empírico pagado

Dada una configuración candidata $\Theta$, el oráculo re-simula $I_p$ iteraciones del anfitrión sobre $K$ anclas retrospectivas $\{A_k\}$ y reporta la mediana del mejor valor alcanzado:

$$\mathcal{M}(\Theta) = \underset{k=1..K,\ r=1..R_p}{\mathrm{med}} \ J(\mathcal{A}, \Theta, I_p;\ A_k, \zeta_{k,r})$$

Coste por consulta: $P = \sum_k |A_k| \cdot I_p \cdot R_p \le K \cdot C \cdot I_p \cdot R_p$. Con $K=3$, $C=20$, $I_p = R_p = 1$: **$P \le 60$ FEs por consulta**. Las semillas $\zeta_{k,r}$ se fijan entre candidatos (*Common Random Numbers*), reduciendo la varianza de las comparaciones:

$$\mathrm{Var}(\mathcal{M}(\Theta) - \mathcal{M}(\Theta')) = \mathrm{Var}(\mathcal{M}(\Theta)) + \mathrm{Var}(\mathcal{M}(\Theta')) - 2\,\mathrm{Cov}(\mathcal{M}(\Theta), \mathcal{M}(\Theta'))$$

El predictor está memoizado: consultas repetidas no se recargan.

### Estimación de importancia (explicadores XAI-CF)

Los cuatro explicadores operan sobre el mismo $\mathcal{M}$ y producen $\bar{\Phi} = \{\phi_1, \dots, \phi_m\}$:

| Explicador | Forma | Cota de consultas $Q_v(m)$ |
|---|---|---|
| SHAP-CF | $\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{\|S\|!\,(m-\|S\|-1)!}{m!} [\mathcal{M}(S \cup \{j\}) - \mathcal{M}(S)]$ | $\lfloor Q_{shap}/(m{+}1) \rfloor (m{+}1)$ |
| LIME-CF | $g^\* = \arg\min_{g} \sum_z \pi_\Theta(z)(\mathcal{M}(z) - g(z))^2 + \Omega(g)$ | $\max(m{+}2, Q_{lime})$ |
| ACME-CF | $\phi_j = \frac{1}{K}\sum_k \frac{\|\mathcal{M}(\Theta_{-j}, \theta_j + \Delta_j; A_k) - \mathcal{M}(\Theta; A_k)\|}{\|\Delta_j\| + \varepsilon}$ | $1 + g\,m$ |
| iBreakdown-CF | $\phi_{\pi_j} = \mathcal{M}(\Theta_{S_j}) - \mathcal{M}(\Theta_{S_{j-1}})$ | $1 + m' + \min(\pi, \binom{m'}{2})$ |

Hiperparámetros de penalización: $Q_{shap}=16$, $Q_{lime}=12$, $g=3$, $\pi=4$, $m' = \min(m, 6)$. Peor caso por activación: 1140 FEs (BA con ACME-CF), es decir 0,57 % de $MaxFEs$ en $D=10$ y 0,114 % en $D=20$.

### Síntesis contrafactual determinista

La formulación contrafactual clásica restringida al espacio factible sería:

$$\Theta^{cf} = \arg\min_{\Theta' \in \Omega_\Theta} \left[ \mathcal{M}(\Theta') + \lambda \lVert \Theta' - \Theta_t \rVert_1 \right]$$

Para evitar esa optimización interna costosa, X-Opt la reemplaza por una prescripción determinista, acotada y sin FEs adicionales:

$$\theta_j^{cf} = \Pi_{[L_j, U_j]}\!\left( \theta_j + d_j \cdot S \cdot \frac{|\phi_j|}{\sum_k |\phi_k|} \cdot (U_j - L_j) \right)$$

donde $\Pi$ proyecta al intervalo permitido, $d_j \in \{-1, +1\}$ es la dirección (inferida desde configuraciones exitosas recientes, con fallback determinista) y $S = 0{,}1$ la intensidad máxima. **El explicador decide qué mover; el contrafactual decide cuánto y hacia dónde.**

### Cotas de actividad y complejidad

**Proposición 2.** El número de activaciones por corrida satisface $A \le \lceil MaxFEs / W_{cd} \rceil$ (= 100 con $W_{cd} = 0{,}01 \cdot MaxFEs$) y la fracción de presupuesto del oráculo cumple $FE^{exp}/MaxFEs \le A \cdot \max_v R_v(m) / MaxFEs$.

Tiempo total de una corrida asistida:

$$T = O(MaxFEs \cdot (D + c_f)) \cdot \left(1 + O\!\left(\tfrac{A \cdot R_v(m)}{MaxFEs}\right)\right)$$

La asistencia no altera la clase asintótica del anfitrión; añade un factor multiplicativo acotado. Memoria: buffer de $B = 64$ instantáneas, $O(B \cdot N \cdot D)$; la huella conserva el orden $O(N \cdot D)$ del anfitrión.

---

## Pseudocódigo completo

```text
ENTRADA: anfitrión A, función objetivo f, presupuesto MaxFEs,
         explicador v ∈ {SHAP, LIME, ACME, iBreakdown},
         parámetros accionables Θ con límites [Lj, Uj]

Inicializar población P0, fitness F0, parámetros basales Θ0
Inicializar ledger B: FE_opt = FE_exp = FE_dir = FE_cf = 0
Inicializar historial retrospectivo W_hist, cooldown t_cd = 0

MIENTRAS FE_tot(B) < MaxFEs:
    SI |f(x_best) − f*| < 1e−8:                # parada oficial CEC
        registrar(ObjetivoAlcanzado); ROMPER

    # ---- SUPERVISOR X-OPT (Algoritmo 1) ----
    SI I_stag(t) = 1 Y v ≠ Standard:
        {A_k} ← AnclasRetrospectivas(W_hist, K=3, C=20)
        R ← Q_v(m) · Σ_k |A_k| · I_p · R_p     # pre-reserva
        SI MaxFEs − FE_tot(B) < R:
            registrar(OmitidoPorDisponibilidad)
            cooldown ← W_cd / 3
        SI NO:
            Φ̄  ← Explicar_v(M, {A_k}, Θ_t)     # cada sonda carga FE_exp
            d  ← InferirDirección(historial H)  # sin coste en FEs
            Θcf ← Π_[L,U]( Θ_t + d·S·(|Φ̄|/Σ|Φ̄|)·(U−L) )
            AplicarIntervención(Θcf)            # override persistente
            cooldown ← W_cd
            registrar(FE_exp, Φ̄, Θ_t, Θcf, σ²(P_t))
            programar auditoría en t + W_post

    # ---- CICLO BASAL DEL ANFITRIÓN ----
    generar una generación con operadores nativos de A usando Θ_t
    evaluar candidatos y cargar como FE_opt
    actualizar población, fitness, x_best y parámetros activos
    registrar diversidad σ²(P_t); guardar snapshot en W_hist
    ejecutar auditorías post-hoc vencidas → {rescate | neutro | interferencia}

    # ---- ADAPTADOR DE INTERVENCIÓN (si hay override activo) ----
    SI anfitrión con memoria (L-SHADE, jSO):    # Algoritmo 3
        proyectar Θcf sobre parámetros activos compatibles
        PRESERVAR memorias M_F, M_CR, archivo externo y reglas endógenas
        si hay reemplazos exitosos → la memoria se actualiza con la
        política propia del anfitrión
    SI NO (PSO, BA, GWO, GSK, OPA, SBOA):       # Algoritmo 4
        Θ_t ← Π_ΩΘ(Θcf)                         # override directo
        vigente hasta nueva activación, objetivo o fin de presupuesto

VERIFICAR: FE_opt + FE_exp + FE_dir + FE_cf = FE_tot ≤ MaxFEs   # aserción
DEVOLVER x_best, ledger B, intervenciones y auditorías
```

---

## Protocolo experimental

| Elemento | Configuración |
|---|---|
| Suite | IEEE CEC 2022, F1–F12 (unimodal, multimodal, híbridas, composición) |
| Dimensiones / presupuesto | $D=10$: $2\times10^5$ FEs · $D=20$: $10^6$ FEs |
| Corridas | 30 por celda, semillas pareadas (CRN) Standard vs. asistidas |
| Anfitriones | PSO, BA, GWO, L-SHADE, jSO, GSK, OPA, SBOA |
| Variantes | Standard, SHAP-CF, LIME-CF, ACME-CF, iBreakdown-CF |
| Métrica / éxito | $\|f(x_{best}) - f^*\|$ · umbral $10^{-8}$ |
| Estadística | Wilcoxon pareado por celda + Holm + Friedman |
| Supervisor | $W_{trig}=0{,}02$, $W_{cd}=0{,}01$, $W_{hist}=0{,}02$ (× MaxFEs), $S=0{,}1$, 1 CF por activación |

**Parámetros accionables por anfitrión** (fuente de verdad: `engines.py`):

| Anfitrión | Parámetros ($m$) | Canal |
|---|---|---|
| PSO | $w$, $c_1$, $c_2$, $v_{max}$ (4) | Override directo |
| BA | $f_{min}$, $f_{max}$, loudness, pulse rate, $\alpha$, $\gamma$ (6) | Override directo |
| GWO | $a_{scale}$, pesos de líder $\alpha/\beta/\delta$ (4) | Override directo |
| GSK | $K_F$, $K_R$ (2) | Override directo |
| OPA | drive, encircle, attack, explore prob (4) | Override directo |
| SBOA | hunt, escape, explore prob, local (4) | Override directo |
| L-SHADE | $F$, $CR$, pbest rate, archive rate (4) | Sensible a memoria ($M_F$, $M_{CR}$ preservadas) |
| jSO | $F$, $CR$, pbest rate, archive rate (4) | Override restringido |

---

## Resultados clave

**Selectividad, no mejora universal.** Sobre 96 celdas por dimensión (mejor variante vs. Standard, umbral ±5 % en mediana):

| | Mejora | Nula | Adversa |
|---|---|---|---|
| $D=10$ | 32 | 57 | 7 |
| $D=20$ | 44 | 49 | 3 |

- **Núcleo receptivo**: SBOA, GSK y BA (Friedman significativo en ambas dimensiones para SBOA y GSK).
- **Frontera**: jSO — mejoras locales amortiguadas por su auto-adaptación.
- **Neutros / resistentes**: PSO, L-SHADE, OPA. **Interferencia puntual**: GWO.
- **Explicadores**: ACME-CF obtiene el mejor rango promedio global (1.65) en anfitriones receptivos; SHAP, LIME e iBreakdown dominan combinaciones específicas.
- **Formas del rescate**: desplazamiento completo de la distribución (SBOA–F1), compresión de mediana e IQR en órdenes de magnitud (BA–F3), aumento de la probabilidad de éxito (GSK–F1, $D=20$: 17/30 corridas exitosas con LIME-CF vs. 1/30 en Standard), reducción amplia sin alcanzar el umbral (jSO–F6).

**Traslado aplicado** — hub de carga de vehículos eléctricos ($D=96$, 24 h multiperiodo, BESS + diésel + límites de red, penalización $f + \rho \cdot CV$ con $\rho = 10^6$; 10 escenarios × 4 anfitriones × 5 variantes × 30 corridas = 6 000 ejecuciones):

- Comparación primaria preregistrada **GSK–ACME-CF**: significativa tras Holm en los **10/10 escenarios**, ganancia pareada mediana +0,310 % (rango +0,075 % a +0,887 %), robusta distributivamente.
- Fuera de ese par, el panorama es mayoritariamente neutro (106/160 comparaciones); 7 favorecen significativamente a Standard.
- La jerarquía de receptividad del benchmark **predice el comportamiento aplicado**, validando el diseño en embudo. El anfitrión que captura beneficio (GSK) es también el que menos presupuesto explicativo consume (0,4–2,2 %), mientras BA y jSO gastan 8–13 % sin retorno.

**Frontera de aplicabilidad**: el estancamiento del objetivo es señal de intervención suficiente solo cuando la factibilidad no domina la dificultad del problema. Trabajo futuro: ablación por componente, controlador con capacidad de abstención y políticas adaptativas de selección del explicador.

---

## Estructura del repositorio

```
.
├── CEC2022 Oficial Format/     # resultados en el formato oficial de la
│                               # competencia IEEE CEC 2022, por anfitrión
├── datos_principal/            # campaña benchmark: logs por corrida, ledger
│                               # de FEs, intervenciones y tablas agregadas
├── datos_aplicado/             # campaña del caso aplicado (hub de carga EV,
│                               # 10 escenarios x 4 anfitriones x 5 variantes)
├── paquetes_repositorio/       # paquetes y artefactos de verificación
├── scripts/                    # código Python: motores de los 8 anfitriones,
│                               # oráculo, ledger, gate, explicadores XAI-CF,
│                               # síntesis contrafactual y análisis estadístico
├── ESTADO_PROYECTO.md
└── README.md
```

El manuscrito LaTeX se mantiene en un proyecto separado y no se
distribuye en este repositorio.
---

## Referencia

Si utilizas este trabajo:

> Olivares, P., Olivares, R. *Marco de trabajo para aproximación de parámetros basado en modelos agnósticos explicables para algoritmos de Optimización Global* (X-Opt Oracle). Universidad de Valparaíso, 2026.## Licencias

- **Código** (`scripts/`): [MIT](LICENSE). Uso, modificación y redistribución libres con atribución — el estándar para investigación reproducible.
- **Datos experimentales** (`CEC2022 Oficial Format/`, `datos_principal/`, `datos_aplicado/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.es). Pueden reutilizarse citando la fuente, que es la licencia habitual para datos de investigación.
- El manuscrito y sus figuras **no forman parte de este repositorio**; todos los derechos sobre ellos quedan reservados hasta la publicación del artículo.

Para citar este trabajo, GitHub genera la cita automáticamente desde [`CITATION.cff`](CITATION.cff) (botón *"Cite this repository"* en la barra lateral del repo).

# X-Opt Oracle

**Marco de trabajo supervisor XAI-contrafactual para el ajuste dinámico de parámetros en algoritmos de optimización global.**

Pablo Olivares, Rodrigo Olivares — Escuela de Ingeniería Informática, Universidad de Valparaíso, Chile.

---

## Resumen

Las metaheurísticas poblacionales suelen estancarse en óptimos locales sin mecanismos que expliquen por qué ocurre ni que orienten su recuperación. **X-Opt Oracle** es una capa supervisora desacoplada que monitorea al algoritmo anfitrión y, ante una señal de estancamiento, ejecuta una secuencia diagnóstica-prescriptiva:

1. Construye un **oráculo empírico pagado** $\mathcal{M}(\Theta)$ que re-simula horizontes cortos de búsqueda, cargando cada evaluación al mismo presupuesto global.
2. Estima la **importancia de los parámetros accionables** mediante un explicador XAI intercambiable (SHAP, LIME, ACME o iBreakdown).
3. Sintetiza una **reconfiguración contrafactual determinista** y la aplica como override persistente.
4. **Audita** el efecto posterior (rescate, neutralidad o interferencia).

Todo el proceso opera bajo un **ledger estricto de evaluaciones** con pre-reserva presupuestaria: ninguna mejora proviene de evaluaciones ocultas.

El hallazgo central es que la asistencia XAI-CF **no produce mejoras universales, sino beneficios selectivos** determinados por la *plasticidad paramétrica* del anfitrión: SBOA, GSK y BA son receptivos; los mecanismos auto-adaptativos de L-SHADE y jSO amortiguan la intervención; PSO permanece neutro; GWO y OPA delimitan la zona de interferencia. ACME-CF es el explicador con mejor rango global entre los anfitriones receptivos. En el traslado aplicado (planificación de un hub de carga de vehículos eléctricos), GSK–ACME-CF supera estadísticamente a Standard en los diez escenarios evaluados, con mejoras de magnitud moderada.

---

## Contribuciones

- La explicabilidad deja de ser inspección post-hoc y se convierte en **componente operativo de control de lazo cerrado**.
- Arquitectura **agnóstica al explicador y al anfitrión**: 8 metaheurísticas × 4 explicadores bajo la misma contabilidad.
- **Auditabilidad formal**: ledger con invariante de suma exacta (Proposición 1) y cota de actividad del oráculo (Proposición 2).
- Caracterización empírica de la **frontera de aplicabilidad**: cuándo la intervención rescata, cuándo es neutra y cuándo interfiere.

---

## Matemáticas del oráculo

### Problema y estancamiento

Optimización global continua sobre $\Omega = \prod_{j=1}^{D}[L_j, U_j]$:

$$\min_{x \in \Omega} f(x)$$

Sea $\tau^\star(t)$ la última evaluación con mejora relativa del incumbente superior a $\delta = 10^{-8}$ (considerando solo evaluaciones del optimizador). El disparador de estancamiento es:

$$
I_{stag}(t) =
\begin{cases}
1 & \text{si } t - \tau^\star(t) \ge W_{trig} \ \wedge \ t \ge t_{cd} \\
0 & \text{en otro caso}
\end{cases}
\qquad W_{trig} = \lceil 0{,}02 \cdot MaxFEs \rceil
$$

con cooldown $W_{cd} = \lceil 0{,}01 \cdot MaxFEs \rceil$ tras intervenir ($W_{cd}/3$ tras un intento bloqueado). La diversidad poblacional se registra como telemetría:

$$\sigma^2(P_t) = \frac{1}{N \cdot D} \sum_{i=1}^{N} \sum_{j=1}^{D} (x_{i,j} - \bar{x}_j)^2$$

### Ledger estricto y gate de disponibilidad

Toda evaluación real se imputa a una de cuatro categorías:

$$FE^{tot}(t) = FE^{opt}(t) + FE^{exp}(t) + FE^{dir}(t) + FE^{cf}(t) \le MaxFEs$$

(en la implementación evaluada $FE^{dir} = FE^{cf} = 0$; la dirección se infiere sin coste desde el historial). Antes de activar el explicador $v$ sobre $m$ parámetros, el gate exige la pre-reserva:

$$MaxFEs - FE^{tot}(t) \ \ge\ R_v(m) = Q_v(m) \cdot P$$

**Proposición 1 (seguridad presupuestaria).** Bajo el ledger con pre-reserva, toda corrida satisface $FE^{tot} \le MaxFEs$ y la suma de categorías coincide exactamente con $FE^{tot}$ (verificado con aserción al cierre de cada corrida).

### Oráculo empírico pagado

Dada una configuración candidata $\Theta$, el oráculo re-simula $I_p$ iteraciones del anfitrión sobre $K$ anclas retrospectivas $\{A_k\}$ y reporta la mediana del mejor valor alcanzado:

$$\mathcal{M}(\Theta) = \underset{k=1..K,\ r=1..R_p}{\mathrm{med}} \ J(\mathcal{A}, \Theta, I_p;\ A_k, \zeta_{k,r})$$

Coste por consulta: $P = \sum_k |A_k| \cdot I_p \cdot R_p \le K \cdot C \cdot I_p \cdot R_p$. Con $K=3$, $C=20$, $I_p = R_p = 1$: **$P \le 60$ FEs por consulta**. Las semillas $\zeta_{k,r}$ se fijan entre candidatos (*Common Random Numbers*), reduciendo la varianza de las comparaciones:

$$\mathrm{Var}(\mathcal{M}(\Theta) - \mathcal{M}(\Theta')) = \mathrm{Var}(\mathcal{M}(\Theta)) + \mathrm{Var}(\mathcal{M}(\Theta')) - 2\,\mathrm{Cov}(\mathcal{M}(\Theta), \mathcal{M}(\Theta'))$$

El predictor está memoizado: consultas repetidas no se recargan.

### Estimación de importancia (explicadores XAI-CF)

Los cuatro explicadores operan sobre el mismo $\mathcal{M}$ y producen $\bar{\Phi} = \{\phi_1, \dots, \phi_m\}$:

| Explicador | Forma | Cota de consultas $Q_v(m)$ |
|---|---|---|
| SHAP-CF | $\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{\|S\|!\,(m-\|S\|-1)!}{m!} [\mathcal{M}(S \cup \{j\}) - \mathcal{M}(S)]$ | $\lfloor Q_{shap}/(m{+}1) \rfloor (m{+}1)$ |
| LIME-CF | $g^\* = \arg\min_{g} \sum_z \pi_\Theta(z)(\mathcal{M}(z) - g(z))^2 + \Omega(g)$ | $\max(m{+}2, Q_{lime})$ |
| ACME-CF | $\phi_j = \frac{1}{K}\sum_k \frac{\|\mathcal{M}(\Theta_{-j}, \theta_j + \Delta_j; A_k) - \mathcal{M}(\Theta; A_k)\|}{\|\Delta_j\| + \varepsilon}$ | $1 + g\,m$ |
| iBreakdown-CF | $\phi_{\pi_j} = \mathcal{M}(\Theta_{S_j}) - \mathcal{M}(\Theta_{S_{j-1}})$ | $1 + m' + \min(\pi, \binom{m'}{2})$ |

Hiperparámetros de penalización: $Q_{shap}=16$, $Q_{lime}=12$, $g=3$, $\pi=4$, $m' = \min(m, 6)$. Peor caso por activación: 1140 FEs (BA con ACME-CF), es decir 0,57 % de $MaxFEs$ en $D=10$ y 0,114 % en $D=20$.

### Síntesis contrafactual determinista

La formulación contrafactual clásica restringida al espacio factible sería:

$$\Theta^{cf} = \arg\min_{\Theta' \in \Omega_\Theta} \left[ \mathcal{M}(\Theta') + \lambda \lVert \Theta' - \Theta_t \rVert_1 \right]$$

Para evitar esa optimización interna costosa, X-Opt la reemplaza por una prescripción determinista, acotada y sin FEs adicionales:

$$\theta_j^{cf} = \Pi_{[L_j, U_j]}\!\left( \theta_j + d_j \cdot S \cdot \frac{|\phi_j|}{\sum_k |\phi_k|} \cdot (U_j - L_j) \right)$$

donde $\Pi$ proyecta al intervalo permitido, $d_j \in \{-1, +1\}$ es la dirección (inferida desde configuraciones exitosas recientes, con fallback determinista) y $S = 0{,}1$ la intensidad máxima. **El explicador decide qué mover; el contrafactual decide cuánto y hacia dónde.**

### Cotas de actividad y complejidad

**Proposición 2.** El número de activaciones por corrida satisface $A \le \lceil MaxFEs / W_{cd} \rceil$ (= 100 con $W_{cd} = 0{,}01 \cdot MaxFEs$) y la fracción de presupuesto del oráculo cumple $FE^{exp}/MaxFEs \le A \cdot \max_v R_v(m) / MaxFEs$.

Tiempo total de una corrida asistida:

$$T = O(MaxFEs \cdot (D + c_f)) \cdot \left(1 + O\!\left(\tfrac{A \cdot R_v(m)}{MaxFEs}\right)\right)$$

La asistencia no altera la clase asintótica del anfitrión; añade un factor multiplicativo acotado. Memoria: buffer de $B = 64$ instantáneas, $O(B \cdot N \cdot D)$; la huella conserva el orden $O(N \cdot D)$ del anfitrión.

---

## Pseudocódigo completo

```text
ENTRADA: anfitrión A, función objetivo f, presupuesto MaxFEs,
         explicador v ∈ {SHAP, LIME, ACME, iBreakdown},
         parámetros accionables Θ con límites [Lj, Uj]

Inicializar población P0, fitness F0, parámetros basales Θ0
Inicializar ledger B: FE_opt = FE_exp = FE_dir = FE_cf = 0
Inicializar historial retrospectivo W_hist, cooldown t_cd = 0

MIENTRAS FE_tot(B) < MaxFEs:
    SI |f(x_best) − f*| < 1e−8:                # parada oficial CEC
        registrar(ObjetivoAlcanzado); ROMPER

    # ---- SUPERVISOR X-OPT (Algoritmo 1) ----
    SI I_stag(t) = 1 Y v ≠ Standard:
        {A_k} ← AnclasRetrospectivas(W_hist, K=3, C=20)
        R ← Q_v(m) · Σ_k |A_k| · I_p · R_p     # pre-reserva
        SI MaxFEs − FE_tot(B) < R:
            registrar(OmitidoPorDisponibilidad)
            cooldown ← W_cd / 3
        SI NO:
            Φ̄  ← Explicar_v(M, {A_k}, Θ_t)     # cada sonda carga FE_exp
            d  ← InferirDirección(historial H)  # sin coste en FEs
            Θcf ← Π_[L,U]( Θ_t + d·S·(|Φ̄|/Σ|Φ̄|)·(U−L) )
            AplicarIntervención(Θcf)            # override persistente
            cooldown ← W_cd
            registrar(FE_exp, Φ̄, Θ_t, Θcf, σ²(P_t))
            programar auditoría en t + W_post

    # ---- CICLO BASAL DEL ANFITRIÓN ----
    generar una generación con operadores nativos de A usando Θ_t
    evaluar candidatos y cargar como FE_opt
    actualizar población, fitness, x_best y parámetros activos
    registrar diversidad σ²(P_t); guardar snapshot en W_hist
    ejecutar auditorías post-hoc vencidas → {rescate | neutro | interferencia}

    # ---- ADAPTADOR DE INTERVENCIÓN (si hay override activo) ----
    SI anfitrión con memoria (L-SHADE, jSO):    # Algoritmo 3
        proyectar Θcf sobre parámetros activos compatibles
        PRESERVAR memorias M_F, M_CR, archivo externo y reglas endógenas
        si hay reemplazos exitosos → la memoria se actualiza con la
        política propia del anfitrión
    SI NO (PSO, BA, GWO, GSK, OPA, SBOA):       # Algoritmo 4
        Θ_t ← Π_ΩΘ(Θcf)                         # override directo
        vigente hasta nueva activación, objetivo o fin de presupuesto

VERIFICAR: FE_opt + FE_exp + FE_dir + FE_cf = FE_tot ≤ MaxFEs   # aserción
DEVOLVER x_best, ledger B, intervenciones y auditorías
```

---

## Protocolo experimental

| Elemento | Configuración |
|---|---|
| Suite | IEEE CEC 2022, F1–F12 (unimodal, multimodal, híbridas, composición) |
| Dimensiones / presupuesto | $D=10$: $2\times10^5$ FEs · $D=20$: $10^6$ FEs |
| Corridas | 30 por celda, semillas pareadas (CRN) Standard vs. asistidas |
| Anfitriones | PSO, BA, GWO, L-SHADE, jSO, GSK, OPA, SBOA |
| Variantes | Standard, SHAP-CF, LIME-CF, ACME-CF, iBreakdown-CF |
| Métrica / éxito | $\|f(x_{best}) - f^*\|$ · umbral $10^{-8}$ |
| Estadística | Wilcoxon pareado por celda + Holm + Friedman |
| Supervisor | $W_{trig}=0{,}02$, $W_{cd}=0{,}01$, $W_{hist}=0{,}02$ (× MaxFEs), $S=0{,}1$, 1 CF por activación |

**Parámetros accionables por anfitrión** (fuente de verdad: `engines.py`):

| Anfitrión | Parámetros ($m$) | Canal |
|---|---|---|
| PSO | $w$, $c_1$, $c_2$, $v_{max}$ (4) | Override directo |
| BA | $f_{min}$, $f_{max}$, loudness, pulse rate, $\alpha$, $\gamma$ (6) | Override directo |
| GWO | $a_{scale}$, pesos de líder $\alpha/\beta/\delta$ (4) | Override directo |
| GSK | $K_F$, $K_R$ (2) | Override directo |
| OPA | drive, encircle, attack, explore prob (4) | Override directo |
| SBOA | hunt, escape, explore prob, local (4) | Override directo |
| L-SHADE | $F$, $CR$, pbest rate, archive rate (4) | Sensible a memoria ($M_F$, $M_{CR}$ preservadas) |
| jSO | $F$, $CR$, pbest rate, archive rate (4) | Override restringido |

---

## Resultados clave

**Selectividad, no mejora universal.** Sobre 96 celdas por dimensión (mejor variante vs. Standard, umbral ±5 % en mediana):

| | Mejora | Nula | Adversa |
|---|---|---|---|
| $D=10$ | 32 | 57 | 7 |
| $D=20$ | 44 | 49 | 3 |

- **Núcleo receptivo**: SBOA, GSK y BA (Friedman significativo en ambas dimensiones para SBOA y GSK).
- **Frontera**: jSO — mejoras locales amortiguadas por su auto-adaptación.
- **Neutros / resistentes**: PSO, L-SHADE, OPA. **Interferencia puntual**: GWO.
- **Explicadores**: ACME-CF obtiene el mejor rango promedio global (1.65) en anfitriones receptivos; SHAP, LIME e iBreakdown dominan combinaciones específicas.
- **Formas del rescate**: desplazamiento completo de la distribución (SBOA–F1), compresión de mediana e IQR en órdenes de magnitud (BA–F3), aumento de la probabilidad de éxito (GSK–F1, $D=20$: 17/30 corridas exitosas con LIME-CF vs. 1/30 en Standard), reducción amplia sin alcanzar el umbral (jSO–F6).

**Traslado aplicado** — hub de carga de vehículos eléctricos ($D=96$, 24 h multiperiodo, BESS + diésel + límites de red, penalización $f + \rho \cdot CV$ con $\rho = 10^6$; 10 escenarios × 4 anfitriones × 5 variantes × 30 corridas = 6 000 ejecuciones):

- Comparación primaria preregistrada **GSK–ACME-CF**: significativa tras Holm en los **10/10 escenarios**, ganancia pareada mediana +0,310 % (rango +0,075 % a +0,887 %), robusta distributivamente.
- Fuera de ese par, el panorama es mayoritariamente neutro (106/160 comparaciones); 7 favorecen significativamente a Standard.
- La jerarquía de receptividad del benchmark **predice el comportamiento aplicado**, validando el diseño en embudo. El anfitrión que captura beneficio (GSK) es también el que menos presupuesto explicativo consume (0,4–2,2 %), mientras BA y jSO gastan 8–13 % sin retorno.

**Frontera de aplicabilidad**: el estancamiento del objetivo es señal de intervención suficiente solo cuando la factibilidad no domina la dificultad del problema. Trabajo futuro: ablación por componente, controlador con capacidad de abstención y políticas adaptativas de selección del explicador.

---

## Estructura del repositorio

```
.
├── CEC2022 Oficial Format/     # resultados en el formato oficial de la
│                               # competencia IEEE CEC 2022, por anfitrión
├── datos_principal/            # campaña benchmark: logs por corrida, ledger
│                               # de FEs, intervenciones y tablas agregadas
├── datos_aplicado/             # campaña del caso aplicado (hub de carga EV,
│                               # 10 escenarios x 4 anfitriones x 5 variantes)
├── paquetes_repositorio/       # paquetes y artefactos de verificación
├── scripts/                    # código Python: motores de los 8 anfitriones,
│                               # oráculo, ledger, gate, explicadores XAI-CF,
│                               # síntesis contrafactual y análisis estadístico
├── ESTADO_PROYECTO.md
└── README.md
```

El manuscrito LaTeX se mantiene en un proyecto separado y no se
distribuye en este repositorio.
---

## Referencia

Si utilizas este trabajo:

> Olivares, P., Olivares, R. *Marco de trabajo para aproximación de parámetros basado en modelos agnósticos explicables para algoritmos de Optimización Global* (X-Opt Oracle). Universidad de Valparaíso, 2026.
