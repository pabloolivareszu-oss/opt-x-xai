# X-Opt Oracle

**An XAI-counterfactual supervisory framework for dynamic parameter control in global optimization algorithms**

Pablo Olivares and Rodrigo Olivares  
School of Computer Engineering, Universidad de Valparaíso, Chile

---

## Overview

Population-based metaheuristics may stagnate in local optima without mechanisms that explain why the search has stopped improving or how it should recover. **X-Opt Oracle** is a decoupled supervisory layer that monitors a host optimizer and activates only when a stagnation condition is detected.

Each valid activation performs four steps:

1. It builds a **budget-aware empirical oracle** from short, retrospective search probes.
2. It estimates the contribution of the host's **actionable parameters** using SHAP, LIME, ACME, or iBreakdown.
3. It synthesizes a **deterministic counterfactual configuration**.
4. It applies the configuration and audits the subsequent behavior as rescue, neutrality, or interference.

Every real objective-function evaluation is charged to the same strict evaluation budget. The explanatory stage therefore does not receive hidden evaluations or an auxiliary optimization budget.

The main finding is that XAI-CF assistance is **selective rather than universal**. Its usefulness depends on the parameter plasticity of the host optimizer and on the dominant difficulty of the search landscape. SBOA, GSK, and BA form the receptive core; L-SHADE and jSO tend to damp external intervention through their self-adaptive mechanisms; PSO is mostly neutral; and GWO and OPA help delimit the region in which intervention becomes unstable or adverse. Among receptive hosts, ACME-CF achieves the best overall explanatory ranking.

The framework is evaluated on the IEEE CEC 2022 benchmark and transferred to an applied electric-vehicle charging-hub planning problem.

---

## Main contributions

- Explainability is used as an **operational closed-loop control mechanism**, not only as post-hoc visualization.
- A single framework evaluates eight host optimizers and four XAI-CF explainers under identical accounting rules.
- A strict function-evaluation ledger makes every explanatory evaluation explicit and auditable.
- A budget-availability gate prevents incomplete or unpaid explanatory activations.
- A deterministic counterfactual rule converts parameter importance into a bounded intervention without launching a second optimization process.
- The experiments characterize an empirical **applicability frontier**: rescue, neutrality, and interference.

---

## Framework architecture

X-Opt Oracle is organized as a modular supervisory layer around an unchanged host optimizer.

### Core components

- **Host optimizer**: executes the native population-based search.
- **Telemetry module**: records fitness, diversity, parameters, and function-evaluation consumption.
- **Stagnation monitor**: detects a sustained lack of significant progress.
- **FE Availability Gate**: verifies that the remaining budget can pay for a complete explanatory activation.
- **Empirical oracle**: evaluates candidate parameter configurations through short retrospective probes.
- **XAI explainer**: estimates the local contribution of actionable parameters.
- **Counterfactual generator**: produces a deterministic feasible reconfiguration.
- **Intervention adapter**: injects the configuration into the host.
- **Post-hoc auditor**: classifies the observed outcome.

---

## Mathematical formulation

### Global optimization problem

The continuous global optimization problem is

$$
\min_{x\in\Omega} f(x),
$$

with

$$
\Omega=\prod_{j=1}^{D}[L_j,U_j].
$$

Here, $x$ is a $D$-dimensional decision vector and $[L_j,U_j]$ defines the feasible interval of the $j$-th decision variable.

---

### Stagnation trigger

Let $\tau^\star(t)$ be the most recent evaluation at which the host optimizer produced a relative improvement greater than the threshold $\delta$.

The stagnation indicator is

$$
I_{\mathrm{stag}}(t)=
\begin{cases}
1, & t-\tau^\star(t)\ge W_{\mathrm{trig}}
     \ \land\ t\ge t_{\mathrm{cd}},\\
0, & \text{otherwise}.
\end{cases}
$$

The trigger window is

$$
W_{\mathrm{trig}}
=
\left\lceil
0.02\,\mathrm{MaxFEs}
\right\rceil.
$$

After an accepted intervention, the cooldown is

$$
W_{\mathrm{cd}}
=
\left\lceil
0.01\,\mathrm{MaxFEs}
\right\rceil.
$$

A budget-blocked attempt applies a reduced cooldown of $W_{\mathrm{cd}}/3$.

The experiments use

$$
\delta=10^{-8}.
$$

Population diversity is retained as telemetry:

$$
\sigma^2(P_t)
=
\frac{1}{ND}
\sum_{i=1}^{N}
\sum_{j=1}^{D}
\left(x_{i,j}-\bar{x}_j\right)^2.
$$

Diversity is used to interpret the search dynamics and the effect of intervention. The operational trigger is based on significant progress of the host optimizer.

---

### Strict function-evaluation ledger

Every real objective-function evaluation belongs to exactly one category:

$$
FE^{\mathrm{tot}}(t)
=
FE^{\mathrm{opt}}(t)
+
FE^{\mathrm{exp}}(t)
+
FE^{\mathrm{dir}}(t)
+
FE^{\mathrm{cf}}(t).
$$

The global invariant is

$$
FE^{\mathrm{tot}}(t)
\le
\mathrm{MaxFEs}.
$$

The categories are:

- $FE^{\mathrm{opt}}$: evaluations performed by the native host search.
- $FE^{\mathrm{exp}}$: evaluations performed by the empirical explanatory oracle.
- $FE^{\mathrm{dir}}$: optional evaluations used to infer an intervention direction.
- $FE^{\mathrm{cf}}$: optional evaluations used to validate a counterfactual configuration.

In the evaluated implementation,

$$
FE^{\mathrm{dir}}=0
\qquad\text{and}\qquad
FE^{\mathrm{cf}}=0.
$$

The intervention direction is inferred from the execution history, and the deterministic counterfactual is applied without additional validation evaluations.

**Proposition 1 — Budget safety.**  
Under the strict ledger and the availability pre-reservation rule, every run satisfies the global evaluation budget. At the end of each run, an assertion verifies that the category totals exactly match the global counter.

---

### FE Availability Gate

Before activating explainer $v$ over $m$ actionable parameters, the framework reserves the worst-case cost of the activation:

$$
\mathrm{MaxFEs}-FE^{\mathrm{tot}}(t)
\ge
R_v(m).
$$

The required reservation is

$$
R_v(m)=Q_v(m)\,P,
$$

where:

- $Q_v(m)$ is the maximum number of empirical-oracle queries allowed for explainer $v$.
- $P$ is the worst-case function-evaluation cost of one empirical-oracle query.

When the remaining budget is smaller than $R_v(m)$, the activation is skipped and no explanatory evaluations are consumed.

---

### Budget-aware empirical oracle

For a candidate parameter configuration $\Theta$, the empirical oracle re-simulates a short search horizon over retrospective anchors and reports the median best value:

$$
\mathcal{M}(\Theta)
=
\mathrm{median}_{\substack{
k=1,\ldots,K\\
r=1,\ldots,R_p
}}
J\!\left(
\mathcal{A},
\Theta,
I_p;
A_k,
\zeta_{k,r}
\right).
$$

The notation is:

- $\mathcal{A}$: host optimizer.
- $\Theta$: candidate parameter configuration.
- $I_p$: number of probe iterations.
- $A_k$: retrospective population anchor.
- $\zeta_{k,r}$: random seed used in probe repetition $r$.

The cost of one empirical-oracle query is

$$
P
=
\sum_{k=1}^{K}
\lvert A_k\rvert
I_p
R_p.
$$

The upper bound is

$$
P
\le
KCI_pR_p,
$$

where $C$ is the maximum number of individuals retained per anchor.

For the evaluated configuration,

$$
K=3,\qquad
C=20,\qquad
I_p=1,\qquad
R_p=1,
$$

and therefore

$$
P\le60
$$

function evaluations per empirical-oracle query.

The predictor is memoized. Repeated candidate configurations within the same activation are not evaluated twice.

---

### Common Random Numbers

Candidate parameter configurations are compared using the same probe seeds. This induces positive covariance and reduces the variance of pairwise differences:

$$
\mathrm{Var}
\left[
\mathcal{M}(\Theta)-\mathcal{M}(\Theta')
\right]
=
\mathrm{Var}
\left[
\mathcal{M}(\Theta)
\right]
+
\mathrm{Var}
\left[
\mathcal{M}(\Theta')
\right]
-
2\,
\mathrm{Cov}
\left[
\mathcal{M}(\Theta),
\mathcal{M}(\Theta')
\right].
$$

This design improves local discrimination under a limited explanatory budget.

---

## XAI-CF explainers

All explainers operate on the same empirical oracle and produce a contribution vector

$$
\bar{\Phi}
=
\left\{
\phi_1,\ldots,\phi_m
\right\}.
$$

To avoid GitHub rendering failures, the equations are presented outside Markdown tables.

### SHAP-CF

SHAP estimates the marginal contribution of actionable parameter $j$:

$$
\phi_j^{\mathrm{SHAP}}
=
\sum_{S\subseteq F\setminus\{j\}}
\frac{
\lvert S\rvert!
\left(
m-\lvert S\rvert-1
\right)!
}{
m!
}
\left[
\mathcal{M}(S\cup\{j\})
-
\mathcal{M}(S)
\right].
$$

Its query bound is

$$
Q_{\mathrm{SHAP}}(m)
=
\left\lfloor
\frac{Q_{\mathrm{shap}}}{m+1}
\right\rfloor
(m+1).
$$

---

### LIME-CF

LIME fits a locally weighted interpretable surrogate:

$$
g^\star
=
\arg\min_{g}
\left[
\sum_{z}
\pi_{\Theta}(z)
\left(
\mathcal{M}(z)-g(z)
\right)^2
+
\Omega(g)
\right].
$$

Its query bound is

$$
Q_{\mathrm{LIME}}(m)
=
\max
\left(
m+2,
Q_{\mathrm{lime}}
\right).
$$

---

### ACME-CF

ACME estimates a controlled effect along anchored trajectories:

$$
\phi_j^{\mathrm{ACME}}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{
\left|
\mathcal{M}
\left(
\Theta_{-j},
\theta_j+\Delta_j;
A_k
\right)
-
\mathcal{M}
\left(
\Theta;
A_k
\right)
\right|
}{
\left|
\Delta_j
\right|
+
\varepsilon
}.
$$

Its query bound is

$$
Q_{\mathrm{ACME}}(m)
=
1+gm.
$$

---

### iBreakdown-CF

iBreakdown decomposes the empirical prediction sequentially:

$$
\phi_{\pi_j}^{\mathrm{iBD}}
=
\mathcal{M}
\left(
\Theta_{S_j}
\right)
-
\mathcal{M}
\left(
\Theta_{S_{j-1}}
\right).
$$

Its query bound is

$$
Q_{\mathrm{iBD}}(m)
=
1
+
m'
+
\min
\left(
\pi,
\frac{m'(m'-1)}{2}
\right),
$$

with

$$
m'=\min(m,6).
$$

---

### Explanatory-budget hyperparameters

The evaluated values are

$$
Q_{\mathrm{shap}}=16,
\qquad
Q_{\mathrm{lime}}=12,
\qquad
g=3,
\qquad
\pi=4.
$$

The worst-case cost per activation is 1,140 function evaluations for BA with ACME-CF.

This corresponds to:

- 0.57% of the total budget for $D=10$.
- 0.114% of the total budget for $D=20$.

---

## Deterministic counterfactual synthesis

A classical feasible counterfactual problem can be written as

$$
\Theta^{\mathrm{cf}}
=
\arg\min_{\Theta'\in\Omega_\Theta}
\left[
\mathcal{M}(\Theta')
+
\lambda
\left\|
\Theta'-\Theta_t
\right\|_1
\right].
$$

X-Opt does not solve this nested optimization problem. Instead, it uses the following deterministic prescription:

$$
\theta_j^{\mathrm{cf}}
=
\Pi_{[L_j,U_j]}
\left[
\theta_j
+
d_j
S
\frac{
\left|
\phi_j
\right|
}{
\sum_{k=1}^{m}
\left|
\phi_k
\right|
+
\varepsilon
}
\left(
U_j-L_j
\right)
\right].
$$

Here:

- $\Pi_{[L_j,U_j]}$ projects the result onto the feasible parameter interval.
- $d_j$ belongs to $\{-1,+1\}$ and represents the preferred direction.
- $S$ is the maximum intervention intensity.
- $\varepsilon$ prevents division by zero.

The evaluated intensity is

$$
S=0.1.
$$

> The explainer determines **which parameters should move**.  
> The counterfactual rule determines **how far and in which direction**.

---

## Oracle activity and computational complexity

**Proposition 2 — Oracle activity bound.**  
The number of accepted activations per run satisfies

$$
A
\le
\left\lceil
\frac{
\mathrm{MaxFEs}
}{
W_{\mathrm{cd}}
}
\right\rceil.
$$

With

$$
W_{\mathrm{cd}}
=
0.01\,\mathrm{MaxFEs},
$$

the loose theoretical bound is

$$
A\le100.
$$

The explanatory-budget fraction satisfies

$$
\frac{
FE^{\mathrm{exp}}
}{
\mathrm{MaxFEs}
}
\le
\frac{
A
\max_v
R_v(m)
}{
\mathrm{MaxFEs}
}.
$$

The total time complexity can be expressed as

$$
T
=
O
\left(
\mathrm{MaxFEs}
\left(
D+c_f
\right)
\right)
\left[
1
+
O
\left(
\frac{
A R_v(m)
}{
\mathrm{MaxFEs}
}
\right)
\right].
$$

The supervisory layer does not change the asymptotic complexity class of the host optimizer. It introduces a bounded multiplicative factor.

The supervisor stores a fixed buffer of population snapshots:

$$
O(BND),
$$

with $B=64$. Up to constant factors, the memory order remains $O(ND)$.

---

## Complete pseudocode

```text
INPUT:
    host optimizer A
    objective function f
    evaluation budget MaxFEs
    explainer v in {SHAP, LIME, ACME, iBreakdown}
    actionable parameter vector Theta
    feasible parameter bounds [L, U]

INITIALIZE:
    host population P
    host fitness values F
    baseline parameters Theta_0

    ledger:
        FE_opt = 0
        FE_exp = 0
        FE_dir = 0
        FE_cf  = 0
        FE_tot = 0

    retrospective history W_hist
    successful-configuration history H
    cooldown endpoint t_cd = 0
    pending post-hoc audits = empty

WHILE FE_tot < MaxFEs:

    IF official target has been reached:
        record TargetReached
        BREAK

    update stagnation state using host-only progress

    IF stagnation is active AND variant is not Standard:

        anchors = select retrospective anchors from W_hist
        required_budget = query_bound(v, m) * probe_cost(anchors)

        IF remaining_budget < required_budget:
            record SkippedByAvailability
            set reduced cooldown
        ELSE:
            Phi = explain empirical_oracle using v
            direction = infer_direction(H)
            Theta_cf = deterministic_counterfactual(
                Theta_current,
                Phi,
                direction,
                bounds
            )

            apply Theta_cf through the host adapter
            set normal cooldown

            record:
                explanatory evaluations
                importance vector
                previous parameters
                counterfactual parameters
                current diversity

            schedule a post-hoc audit

    execute one native generation of the host optimizer

    FOR each objective-function evaluation:
        charge one unit to FE_opt
        charge one unit to FE_tot

    update:
        population
        fitness
        incumbent
        active parameters
        diversity
        retrospective history

    execute due post-hoc audits

    IF a counterfactual override is active:

        IF host is L-SHADE or jSO:
            preserve native memories and archive
            inject only compatible active parameters
            allow the host to update its memories
            through its native success-history policy
        ELSE:
            keep the direct parameter override active
            until a new intervention, target attainment,
            or budget exhaustion

ASSERT:
    FE_opt + FE_exp + FE_dir + FE_cf == FE_tot
    FE_tot <= MaxFEs

RETURN:
    best solution
    final ledger
    intervention records
    post-hoc audit records
```

---

## Experimental protocol

| Element | Configuration |
|---|---|
| Benchmark | IEEE CEC 2022, functions F1 to F12 |
| Dimensions | 10 and 20 |
| Evaluation budget for D = 10 | 200,000 FEs |
| Evaluation budget for D = 20 | 1,000,000 FEs |
| Independent runs | 30 per algorithm-variant-function-dimension cell |
| Seed design | Paired seeds through Common Random Numbers |
| Host optimizers | PSO, BA, GWO, L-SHADE, jSO, GSK, OPA, SBOA |
| Variants | Standard, SHAP-CF, LIME-CF, ACME-CF, iBreakdown-CF |
| Primary metric | Absolute final error |
| Success threshold | 1e-8 |
| Statistical tests | Paired Wilcoxon, Holm correction, Friedman test |
| Counterfactuals per activation | 1 |
| Intervention intensity | 0.1 |

---

## Actionable parameters by host

The implementation in `scripts/engines.py` is the source of truth.

| Host | Actionable parameters | Coupling mechanism |
|---|---|---|
| PSO | w, c1, c2, vmax | Direct override |
| BA | fmin, fmax, loudness, pulse rate, alpha, gamma | Direct override |
| GWO | a-scale and leader weights | Direct override |
| GSK | KF and KR | Direct override |
| OPA | drive, encircle, attack, exploration probability | Direct override |
| SBOA | hunt, escape, exploration probability, local-search weight | Direct override |
| L-SHADE | F, CR, p-best rate, archive rate | Memory-sensitive coupling |
| jSO | F, CR, p-best rate, archive rate | Restricted memory-sensitive coupling |

For L-SHADE and jSO, the counterfactual intervention does not destructively replace the native success-history memories. It changes compatible active parameters while preserving the host's internal adaptation mechanism.

---

## Main benchmark results

### Receptivity across CEC 2022

The best XAI-CF variant is compared with Standard using a median-improvement threshold of 5%.

| Dimension | Improved cells | Neutral cells | Adverse cells |
|---|---:|---:|---:|
| D = 10 | 32 | 57 | 7 |
| D = 20 | 44 | 49 | 3 |

The results support the following empirical classification:

- **Receptive core**: SBOA, GSK, and BA.
- **Boundary case**: jSO.
- **Mostly neutral or resistant**: PSO, L-SHADE, and OPA.
- **Localized interference**: GWO.

Among receptive hosts, ACME-CF obtains the best global average rank:

$$
1.65.
$$

SHAP-CF, LIME-CF, and iBreakdown-CF still dominate specific host-function combinations.

The observed rescue patterns include:

- displacement of the final-error distribution;
- reductions in median error and interquartile range;
- increased probability of reaching the target;
- renewed convergence after a long plateau;
- broad error reduction without necessarily reaching the official success threshold.

---

## Applied transfer study

The applied study addresses the planning of an electric-vehicle charging hub with:

- 96 decision variables;
- a 24-hour multiperiod horizon;
- battery energy storage;
- diesel generation;
- grid-import limits;
- penalized constraint handling;
- ten operating scenarios.

The penalized objective is

$$
f_{\mathrm{pen}}
=
f
+
\rho\,CV,
$$

with

$$
\rho=10^6.
$$

The campaign contains

$$
10
\times
4
\times
5
\times
30
=
6000
$$

independent executions.

The preregistered comparison between GSK–ACME-CF and Standard is significant after Holm correction in all ten evaluated scenarios.

The paired median gain is 0.310%, with a range from 0.075% to 0.887%.

Outside the primary comparison:

- most comparisons are neutral;
- 106 of 160 comparisons are classified as neutral;
- seven comparisons significantly favor Standard;
- GSK obtains the most consistent benefit while using the smallest explanatory-budget fraction;
- BA and jSO consume a larger explanatory-budget fraction without comparable returns.

These results support the transfer of the host-receptivity concept from the synthetic benchmark to the applied problem.

---

## Applicability frontier

X-Opt Oracle is most useful when three conditions coincide:

1. The host exposes parameters with sufficient runtime plasticity.
2. The stagnation signal reflects a recoverable loss of search productivity.
3. Feasibility management does not dominate the optimization difficulty.

When these conditions do not hold, the intervention may be neutral or may interfere with the host dynamics.

Planned extensions include:

- component-wise ablation;
- abstention-capable intervention;
- adaptive explainer selection;
- constraint-aware triggering;
- problem-specific intervention adapters.

---

## Repository structure

```text
.
├── CEC2022 Oficial Format/
│   └── Results exported in the official IEEE CEC 2022 format
│
├── datos_principal/
│   └── Main benchmark campaign:
│       per-run logs, FE ledger, interventions, and aggregated results
│
├── datos_aplicado/
│   └── Applied EV charging-hub campaign:
│       scenarios, per-run outputs, and statistical summaries
│
├── paquetes_repositorio/
│   └── Packaged source-code and verification artifacts
│
├── scripts/
│   ├── config.py
│   │   └── Global experimental and supervisory configuration
│   │
│   ├── budget.py
│   │   └── Strict FE ledger and budget-accounting utilities
│   │
│   ├── cec2022_protocol.py
│   │   └── CEC 2022 dimensions, budgets, targets, and stopping rules
│   │
│   ├── engines.py
│   │   └── Host optimizers, actionable parameters, bounds, and adapters
│   │
│   ├── explainers.py
│   │   └── SHAP-CF, LIME-CF, ACME-CF, and iBreakdown-CF
│   │
│   ├── trajectory.py
│   │   └── Retrospective anchors, search history, and telemetry support
│   │
│   ├── utils.py
│   │   └── Shared utilities, validation, serialization, and seed handling
│   │
│   ├── analyze_main_campaign.py
│   │   └── Statistical analysis of the formal CEC 2022 campaign
│   │
│   ├── ev_problem_rwcmop.py
│   │   └── Applied constrained optimization problem definition
│   │
│   ├── run_rwcmop_real.py
│   │   └── Execution entry point for the applied campaign
│   │
│   ├── analyze_rwcmop.py
│   │   └── Applied-campaign statistical analysis
│   │
│   └── plotter_rwcmop.py
│       └── Applied-campaign figures and visual summaries
│
├── CITATION.cff
│   └── Machine-readable citation metadata
│
├── LICENSE
│   └── MIT license for the software
│
├── LICENSE-DATA
│   └── Recommended CC BY 4.0 notice for published datasets
│
└── README.md
    └── Project overview and reproducibility documentation
```

The LaTeX manuscript is maintained separately and is not distributed in this repository.

---

## Citation

When using the software, experimental protocol, or results in academic work, please cite the repository and the associated research.

The repository includes a machine-readable [`CITATION.cff`](CITATION.cff) file. When it is stored at the repository root on GitHub's default branch, GitHub displays a **Cite this repository** option with ready-to-copy citation formats.

A provisional citation is:

> Olivares, P., and Olivares, R. (2026). *X-Opt Oracle: An XAI-counterfactual supervisory framework for dynamic parameter control in global optimization*. Universidad de Valparaíso.

Before publishing the repository, update `CITATION.cff` with:

- the final repository URL;
- the authors' ORCID identifiers, when available;
- the software release version;
- a persistent DOI, when available;
- the final article metadata after publication.

---

## License

### Software

The source code in `scripts/` is distributed under the [MIT License](LICENSE).

The MIT License permits use, copying, modification, distribution, sublicensing, and commercial reuse, provided that the copyright notice and permission notice are retained.

### Experimental data

Unless a dataset directory states otherwise, the experimental outputs are intended for release under the Creative Commons Attribution 4.0 International License.

A separate `LICENSE-DATA` file should be included in the repository root to make the data license explicit.

### Manuscript and figures

The LaTeX manuscript and publication-ready figures are not part of this repository. Their copyright remains reserved until the corresponding publication and reuse terms are established.

---

## Reproducibility requirements

A faithful replication should preserve:

- paired seeds between Standard and assisted variants;
- the strict function-evaluation ledger;
- the FE Availability Gate;
- the official termination criterion;
- the distinction between host and explanatory evaluations;
- the same actionable parameter bounds;
- the same retrospective-anchor policy;
- the same statistical correction procedure.

Changing any of these components modifies the experimental protocol and should be reported explicitly.
