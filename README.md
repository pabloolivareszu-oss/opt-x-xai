# X-Opt Oracle

**An XAI-counterfactual supervisory framework for dynamic parameter control in global optimization algorithms.**

Pablo Olivares, Rodrigo Olivares — School of Computer Engineering, Universidad de Valparaíso, Chile.

---

## Overview

Population-based metaheuristics frequently stagnate in local optima without mechanisms capable of explaining why stagnation occurs or how the search should recover. **X-Opt Oracle** is a decoupled supervisory layer that monitors a host optimizer and, when stagnation is detected, executes a diagnostic-prescriptive sequence:

1. It builds a **budget-aware empirical oracle** $\mathcal{M}(\Theta)$ by re-simulating short search horizons while charging every evaluation to the same global budget.
2. It estimates the **importance of actionable parameters** through an interchangeable XAI explainer: SHAP, LIME, ACME, or iBreakdown.
3. It synthesizes a **deterministic counterfactual reconfiguration** and applies it as a persistent override.
4. It **audits** the subsequent effect as rescue, neutrality, or interference.

The entire process operates under a **strict function-evaluation ledger** with budget pre-reservation. Therefore, no reported improvement can originate from hidden or unaccounted evaluations.

The central finding is that XAI-CF assistance **does not produce universal improvements; it produces selective benefits** determined by the *parameter plasticity* of the host optimizer. SBOA, GSK, and BA are receptive; the self-adaptive mechanisms of L-SHADE and jSO tend to damp external intervention; PSO remains mostly neutral; and GWO and OPA help delimit the interference region. ACME-CF achieves the best overall ranking among receptive hosts. In the applied transfer study involving the planning of an electric-vehicle charging hub, GSK–ACME-CF statistically outperforms the Standard configuration across the ten evaluated scenarios, although the effect sizes remain moderate.

---

## Main contributions

- Explainability is moved beyond post-hoc inspection and becomes an **operational closed-loop control component**.
- The architecture is **host-agnostic and explainer-agnostic**: eight metaheuristics and four explainers are evaluated under the same accounting rules.
- The framework provides **formal auditability** through an exact ledger invariant (Proposition 1) and an upper bound on oracle activity (Proposition 2).
- The experiments characterize an empirical **applicability frontier**, identifying when intervention rescues the search, when it remains neutral, and when it interferes with the host dynamics.

---

## Oracle mathematics

### Global optimization and stagnation

The continuous global optimization problem is defined over

$$
\Omega = \prod_{j=1}^{D}[L_j,U_j]
$$

as

$$
\min_{x \in \Omega} f(x).
$$

Let $\tau^\star(t)$ denote the latest evaluation at which the incumbent produced a relative improvement greater than $\delta = 10^{-8}$, considering only evaluations performed by the host optimizer. The stagnation trigger is

$$
I_{\mathrm{stag}}(t)=
\begin{cases}
1, & \text{if } t-\tau^\star(t)\ge W_{\mathrm{trig}}
     \ \wedge\ t\ge t_{\mathrm{cd}},\\
0, & \text{otherwise},
\end{cases}
\qquad
W_{\mathrm{trig}}=
\left\lceil 0.02\cdot \mathrm{MaxFEs}\right\rceil.
$$

After an intervention, the cooldown is

$$
W_{\mathrm{cd}}=
\left\lceil 0.01\cdot \mathrm{MaxFEs}\right\rceil,
$$

whereas a budget-blocked attempt applies $W_{\mathrm{cd}}/3$.

Population diversity is retained as telemetry:

$$
\sigma^2(P_t)=
\frac{1}{N D}
\sum_{i=1}^{N}
\sum_{j=1}^{D}
\left(x_{i,j}-\bar{x}_j\right)^2.
$$

### Strict ledger and FE Availability Gate

Every real objective-function evaluation is assigned to exactly one category:

$$
FE^{\mathrm{tot}}(t)=
FE^{\mathrm{opt}}(t)+
FE^{\mathrm{exp}}(t)+
FE^{\mathrm{dir}}(t)+
FE^{\mathrm{cf}}(t)
\le \mathrm{MaxFEs}.
$$

In the evaluated implementation,

$$
FE^{\mathrm{dir}}=FE^{\mathrm{cf}}=0,
$$

because the intervention direction is inferred from the execution history without additional objective-function evaluations.

Before activating explainer $v$ over $m$ actionable parameters, the FE Availability Gate requires the following pre-reservation:

$$
\mathrm{MaxFEs}-FE^{\mathrm{tot}}(t)
\ge
R_v(m)=Q_v(m)\cdot P.
$$

**Proposition 1 — Budget safety.** Under the strict ledger and the pre-reservation rule, every run satisfies $FE^{\mathrm{tot}}\le\mathrm{MaxFEs}$, and the sum of all evaluation categories is exactly equal to $FE^{\mathrm{tot}}$. This invariant is verified by an assertion at the end of every run.

### Budget-aware empirical oracle

For a candidate parameter configuration $\Theta$, the empirical oracle re-simulates $I_p$ host iterations over $K$ retrospective anchors $\{A_k\}$ and reports the median best value:

$$
\mathcal{M}(\Theta)=
\underset{k=1,\ldots,K;\ r=1,\ldots,R_p}{\operatorname{median}}
\;
J(\mathcal{A},\Theta,I_p;A_k,\zeta_{k,r}).
$$

The cost of one oracle query is

$$
P=
\sum_{k=1}^{K}|A_k|I_pR_p
\le
KCI_pR_p.
$$

With $K=3$, $C=20$, and $I_p=R_p=1$,

$$
P\le 60 \text{ FEs per query}.
$$

The random seeds $\zeta_{k,r}$ are shared across candidate configurations through *Common Random Numbers* (CRN), reducing the variance of pairwise comparisons:

$$
\operatorname{Var}
\left(
\mathcal{M}(\Theta)-\mathcal{M}(\Theta')
\right)
=
\operatorname{Var}(\mathcal{M}(\Theta))
+
\operatorname{Var}(\mathcal{M}(\Theta'))
-
2\operatorname{Cov}
\left(
\mathcal{M}(\Theta),\mathcal{M}(\Theta')
\right).
$$

The empirical predictor is memoized, so repeated queries are not charged twice.

### XAI-CF importance estimation

All explainers operate on the same empirical oracle $\mathcal{M}$ and return

$$
\bar{\Phi}=\{\phi_1,\ldots,\phi_m\}.
$$

| Explainer | Mathematical form | Query bound $Q_v(m)$ |
|---|---|---|
| SHAP-CF | $\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(m-|S|-1)!}{m!}\left[\mathcal{M}(S\cup\{j\})-\mathcal{M}(S)\right]$ | $\left\lfloor Q_{\mathrm{shap}}/(m+1)\right\rfloor(m+1)$ |
| LIME-CF | $g^\star=\arg\min_g\sum_z\pi_\Theta(z)\left(\mathcal{M}(z)-g(z)\right)^2+\Omega(g)$ | $\max(m+2,Q_{\mathrm{lime}})$ |
| ACME-CF | $\phi_j=\frac{1}{K}\sum_k\frac{\left|\mathcal{M}(\Theta_{-j},\theta_j+\Delta_j;A_k)-\mathcal{M}(\Theta;A_k)\right|}{|\Delta_j|+\varepsilon}$ | $1+gm$ |
| iBreakdown-CF | $\phi_{\pi_j}=\mathcal{M}(\Theta_{S_j})-\mathcal{M}(\Theta_{S_{j-1}})$ | $1+m'+\min\left(\pi,\binom{m'}{2}\right)$ |

The penalty hyperparameters are

$$
Q_{\mathrm{shap}}=16,\qquad
Q_{\mathrm{lime}}=12,\qquad
g=3,\qquad
\pi=4,\qquad
m'=\min(m,6).
$$

The worst-case cost per activation is 1,140 FEs for BA with ACME-CF. This corresponds to 0.57% of $\mathrm{MaxFEs}$ for $D=10$ and 0.114% for $D=20$.

### Deterministic counterfactual synthesis

The classical feasible counterfactual problem can be written as

$$
\Theta^{\mathrm{cf}}
=
\arg\min_{\Theta'\in\Omega_\Theta}
\left[
\mathcal{M}(\Theta')
+
\lambda\|\Theta'-\Theta_t\|_1
\right].
$$

To avoid an expensive nested optimization process, X-Opt replaces it with a deterministic, bounded prescription that requires no additional FEs:

$$
\theta_j^{\mathrm{cf}}
=
\Pi_{[L_j,U_j]}
\left(
\theta_j+
d_jS
\frac{|\phi_j|}{\sum_k|\phi_k|}
(U_j-L_j)
\right),
$$

where $\Pi$ projects the result onto the feasible interval, $d_j\in\{-1,+1\}$ is the preferred direction inferred from recently successful configurations—with a deterministic fallback—and $S=0.1$ is the maximum intervention intensity.

> The explainer determines **what should move**; the counterfactual rule determines **how far and in which direction**.

### Oracle activity and complexity bounds

**Proposition 2 — Oracle activity bound.** The number of activations per run satisfies

$$
A\le
\left\lceil
\frac{\mathrm{MaxFEs}}{W_{\mathrm{cd}}}
\right\rceil.
$$

With $W_{\mathrm{cd}}=0.01\cdot\mathrm{MaxFEs}$, the loose theoretical bound is $A\le100$. The oracle budget fraction satisfies

$$
\frac{FE^{\mathrm{exp}}}{\mathrm{MaxFEs}}
\le
\frac{A\max_vR_v(m)}{\mathrm{MaxFEs}}.
$$

The total running time of an assisted execution is

$$
T=
O\!\left(
\mathrm{MaxFEs}(D+c_f)
\right)
\left[
1+
O\!\left(
\frac{A R_v(m)}{\mathrm{MaxFEs}}
\right)
\right].
$$

Therefore, the assistance layer does not change the asymptotic class of the host optimizer; it introduces a bounded multiplicative factor.

The supervisor stores a buffer of $B=64$ population snapshots, requiring

$$
O(BND)
$$

memory. The overall memory order remains $O(ND)$ up to constant factors.

---

## Complete pseudocode

```text
INPUT:
    host optimizer A
    objective function f
    evaluation budget MaxFEs
    explainer v ∈ {SHAP, LIME, ACME, iBreakdown}
    actionable parameter vector Θ with bounds [Lj, Uj]

INITIALIZE:
    population P0
    fitness values F0
    baseline parameters Θ0
    ledger B:
        FE_opt = FE_exp = FE_dir = FE_cf = 0
    retrospective history W_hist
    cooldown t_cd = 0

WHILE FE_tot(B) < MaxFEs:

    IF |f(x_best) - f*| < 1e-8:
        record(TargetReached)
        BREAK

    # ---------- X-OPT SUPERVISOR ----------
    IF I_stag(t) = 1 AND v != Standard:

        {A_k} <- SelectRetrospectiveAnchors(
                    W_hist,
                    K = 3,
                    C = 20
                 )

        R <- Q_v(m) * SUM_k |A_k| * I_p * R_p

        IF MaxFEs - FE_tot(B) < R:
            record(SkippedByAvailability)
            cooldown <- W_cd / 3

        ELSE:
            Phi <- Explain_v(
                       M,
                       {A_k},
                       Theta_t
                   )

            d <- InferDirection(success_history H)

            Theta_cf <- Project_[L,U](
                            Theta_t
                            + d * S
                            * (|Phi| / SUM |Phi|)
                            * (U - L)
                        )

            ApplyIntervention(Theta_cf)
            cooldown <- W_cd

            record(
                FE_exp,
                Phi,
                Theta_t,
                Theta_cf,
                diversity(P_t)
            )

            schedule post-hoc audit at t + W_post

    # ---------- HOST OPTIMIZER ----------
    generate one native host generation using Theta_t
    evaluate candidates and charge evaluations to FE_opt
    update population, fitness, x_best, and active parameters
    record diversity(P_t)
    save snapshot in W_hist
    execute due post-hoc audits:
        {rescue | neutral | interference}

    # ---------- INTERVENTION ADAPTER ----------
    IF a counterfactual override is active:

        IF host has adaptive memory (L-SHADE or jSO):
            project Theta_cf onto compatible active parameters
            preserve M_F, M_CR, the external archive,
            and the host's endogenous update rules

            IF successful replacements exist:
                update the memories using the host's
                native adaptation policy

        ELSE:
            # PSO, BA, GWO, GSK, OPA, SBOA
            Theta_t <- Project_OmegaTheta(Theta_cf)

            keep the direct override active until:
                a new intervention,
                the target is reached,
                or the evaluation budget is exhausted

ASSERT:
    FE_opt + FE_exp + FE_dir + FE_cf
    = FE_tot
    <= MaxFEs

RETURN:
    x_best
    final ledger B
    intervention log
    audit log
```

---

## Experimental protocol

| Element | Configuration |
|---|---|
| Benchmark | IEEE CEC 2022, F1–F12: unimodal, multimodal, hybrid, and composition functions |
| Dimensions and budgets | $D=10$: $2\times10^5$ FEs; $D=20$: $10^6$ FEs |
| Independent runs | 30 per cell with paired seeds through CRN |
| Host optimizers | PSO, BA, GWO, L-SHADE, jSO, GSK, OPA, and SBOA |
| Variants | Standard, SHAP-CF, LIME-CF, ACME-CF, and iBreakdown-CF |
| Primary metric | $|f(x_{\mathrm{best}})-f^\star|$ |
| Success threshold | $10^{-8}$ |
| Statistical protocol | Paired Wilcoxon tests, Holm correction, and Friedman tests |
| Supervisor settings | $W_{\mathrm{trig}}=0.02$, $W_{\mathrm{cd}}=0.01$, $W_{\mathrm{hist}}=0.02$ as fractions of MaxFEs; $S=0.1$; one counterfactual per activation |

### Actionable parameters by host

The implementation in `engines.py` is the source of truth.

| Host | Actionable parameters | Coupling mechanism |
|---|---|---|
| PSO | $w$, $c_1$, $c_2$, $v_{\max}$ | Direct override |
| BA | $f_{\min}$, $f_{\max}$, loudness, pulse rate, $\alpha$, $\gamma$ | Direct override |
| GWO | $a_{\mathrm{scale}}$ and $\alpha/\beta/\delta$ leader weights | Direct override |
| GSK | $K_F$, $K_R$ | Direct override |
| OPA | drive, encircle, attack, exploration probability | Direct override |
| SBOA | hunt, escape, exploration probability, local-search weight | Direct override |
| L-SHADE | $F$, $CR$, p-best rate, archive rate | Memory-sensitive coupling; $M_F$ and $M_{CR}$ are preserved |
| jSO | $F$, $CR$, p-best rate, archive rate | Restricted memory-sensitive override |

---

## Main results

### Selective rather than universal improvement

Across 96 algorithm-function cells per dimension, comparing the best XAI-CF variant against Standard with a $\pm5\%$ median threshold:

| Dimension | Improved | Neutral | Adverse |
|---|---:|---:|---:|
| $D=10$ | 32 | 57 | 7 |
| $D=20$ | 44 | 49 | 3 |

- **Receptive core:** SBOA, GSK, and BA.
- **Boundary case:** jSO, whose self-adaptation dampens external interventions while preserving local improvements.
- **Neutral or resistant hosts:** PSO, L-SHADE, and OPA.
- **Localized interference:** GWO.
- **Explainer ranking:** ACME-CF achieves the best global average rank, 1.65, among receptive hosts, although SHAP-CF, LIME-CF, and iBreakdown-CF dominate specific host-landscape combinations.
- **Observed rescue patterns:** full distribution shifts, reductions in the median and IQR by several orders of magnitude, increased success probability, and broad error reduction without necessarily reaching the target threshold.

### Applied transfer study

The applied study addresses the planning of an electric-vehicle charging hub with $D=96$, a 24-hour multiperiod model, battery energy storage, diesel generation, grid limits, and penalized constraint handling:

$$
f_{\mathrm{pen}} = f + \rho\,CV,
\qquad
\rho=10^6.
$$

The campaign comprises

$$
10\ \text{scenarios}
\times
4\ \text{hosts}
\times
5\ \text{variants}
\times
30\ \text{runs}
=
6000\ \text{executions}.
$$

Key observations:

- The preregistered **GSK–ACME-CF** comparison remains significant after Holm correction in all ten scenarios.
- Its paired median gain is 0.310%, ranging from 0.075% to 0.887%.
- Outside this primary pair, the applied landscape is mostly neutral: 106 of 160 comparisons are neutral, while seven significantly favor Standard.
- The receptivity hierarchy observed on CEC 2022 predicts the applied behavior.
- GSK captures the most consistent benefit while consuming the smallest explanatory budget fraction, approximately 0.4–2.2%.
- BA and jSO consume approximately 8–13% of the budget in the applied study without comparable returns.

### Applicability frontier

Objective stagnation is a sufficient intervention signal only when feasibility does not dominate the difficulty of the optimization problem. Future work includes component-wise ablation, abstention-capable control, and adaptive explainer selection.

---

## Repository structure

```text
.
├── CEC2022 Oficial Format/     # Results exported in the official IEEE CEC 2022
│                               # competition format, organized by host
├── datos_principal/            # Main benchmark campaign: per-run logs, FE ledger,
│                               # interventions, and aggregated tables
├── datos_aplicado/             # Applied EV charging-hub campaign:
│                               # 10 scenarios x 4 hosts x 5 variants
├── paquetes_repositorio/       # Repository packages and verification artifacts
├── scripts/                    # Python implementation of the eight hosts, empirical
│                               # oracle, ledger, availability gate, XAI-CF explainers,
│                               # deterministic counterfactual synthesis, and analyses
├── ESTADO_PROYECTO.md          # Internal project status and pending tasks
├── CITATION.cff                # Machine-readable citation metadata
├── LICENSE                     # MIT license for the software
└── README.md                   # This document
```

The LaTeX manuscript is maintained in a separate project and is not distributed in this repository.

---

## Citation

When using the software, experimental protocol, or results in academic work, please cite the repository and the associated doctoral research. The repository includes a machine-readable [`CITATION.cff`](CITATION.cff) file. When this file is located in the repository root on the default GitHub branch, GitHub displays a **Cite this repository** option that provides ready-to-copy APA and BibTeX entries.

A provisional human-readable citation is:

> Olivares, P., & Olivares, R. (2026). *X-Opt Oracle: An XAI-counterfactual supervisory framework for dynamic parameter control in global optimization*. Universidad de Valparaíso.

Before making the repository public, update the following metadata in `CITATION.cff`:

- Replace the placeholder repository URL with the final GitHub URL.
- Add the authors' ORCID identifiers when available.
- Add the repository DOI after archiving a release in a persistent repository such as Zenodo.
- Replace the provisional preferred citation with the final article metadata once the paper is published.

---

## License

### Software

The source code in `scripts/` is released under the [MIT License](LICENSE). The license permits use, copying, modification, merging, publication, distribution, sublicensing, and commercial reuse, provided that the copyright and permission notices are retained.

### Experimental data

Unless a dataset directory states otherwise, the experimental outputs in `CEC2022 Oficial Format/`, `datos_principal/`, and `datos_aplicado/` are intended to be released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Reusers must provide appropriate attribution, identify modifications, and link to the license.

For maximum legal clarity, a separate `LICENSE-DATA` file should be included at the repository root when the datasets are published.

### Manuscript and figures

The LaTeX manuscript and publication-ready figures are not part of this repository. Their copyright remains reserved until the corresponding publication terms are established.

---

## Reproducibility note

The repository is intended to support transparent and auditable replication. Any derivative study should preserve the paired-seed design, strict FE ledger, availability pre-reservation, termination criteria, and the distinction between host evaluations and explanatory evaluations. Altering these components changes the experimental protocol and should be reported explicitly.
