#!/usr/bin/env python3
"""
analyze_rwcmop.py
=================
Análisis del experimento RW-CMOP (motor real). Genera, SIN interpretar:
  1) Tabla de estadísticos por problema-algoritmo-variante
     (best, median, mean, std, worst de f; CV media; tasa de factibilidad).
  2) Dinámica del oráculo (activaciones, intervenciones aplicadas, rescates,
     intervenciones útiles) desde per_run.csv.
  3) Tests del protocolo (Wilcoxon pareado Standard-vs-variante, Friedman+Holm),
     APLICADOS SOLO donde hay variación real (si las muestras son idénticas se
     reporta explícitamente, no se fuerza un p-valor).
  4) Curvas de convergencia agregadas (si convergence.csv está disponible).

Salidas: CSV + .tex listos para el paper.
Uso:
    python3 analyze_rwcmop.py --per per_run.csv [--conv convergence.csv] \
                              [--interv interventions.csv] --out analisis_rwcmop
"""
import os, csv, argparse, json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    from scipy.stats import wilcoxon, friedmanchisquare
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ALGS = ["GSK", "SBOA"]
VARIANTS = ["Standard", "SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
XAI = ["SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]


def load(per_path):
    rows = list(csv.DictReader(open(per_path)))
    g = defaultdict(dict)   # (prob,alg,var) -> {run: row}
    for r in rows:
        g[(r["problem"], r["algorithm"], r["variant"])][int(r["run"])] = r
    problems = sorted({r["problem"] for r in rows})
    return rows, g, problems


def fval(r):
    try:
        v = float(r["f"])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def series(group, prob, alg, var):
    """f por run, ordenado por run_id (para pareado)."""
    d = group[(prob, alg, var)]
    runs = sorted(d.keys())
    return np.array([fval(d[rr]) for rr in runs]), runs


def holm(pvals):
    """Corrección de Holm; devuelve p ajustados en el orden de entrada."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]
        prev = max(prev, a)
        adj[idx] = min(1.0, prev)
    return adj


# ---------------------------------------------------------------- 1) TABLA
def build_stats(group, problems, out):
    rows = []
    combos = [(p, a, v) for p in problems for a in ALGS for v in VARIANTS]
    for (prob, alg, var) in tqdm(combos, desc="Estadísticos"):
        d = group[(prob, alg, var)]
        rs = list(d.values())
        fs = np.array([fval(r) for r in rs])
        fs = fs[np.isfinite(fs)]
        feas = 100.0 * sum(int(r["feasible"]) for r in rs) / len(rs)
        cvs = np.array([float(r["cv"]) for r in rs if r["cv"] not in ("", "nan")])
        rows.append({
            "problem": prob, "algorithm": alg, "variant": var,
            "best": np.min(fs) if fs.size else np.nan,
            "median": np.median(fs) if fs.size else np.nan,
            "mean": np.mean(fs) if fs.size else np.nan,
            "std": np.std(fs, ddof=1) if fs.size > 1 else 0.0,
            "worst": np.max(fs) if fs.size else np.nan,
            "cv_mean": np.mean(cvs) if cvs.size else np.nan,
            "feasibility_rate": feas,
        })
    path = os.path.join(out, "stats_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ------------------------------------------------ 2) DINÁMICA DEL ORÁCULO
def build_oracle_dynamics(group, problems, out):
    rows = []
    combos = [(p, a, v) for p in problems for a in ALGS for v in XAI]
    for (prob, alg, var) in tqdm(combos, desc="Dinámica oráculo"):
        rs = list(group[(prob, alg, var)].values())
        act = np.array([float(r["oracle_activations"]) for r in rs])
        app = np.array([float(r["interventions_applied"]) for r in rs])
        res = np.array([float(r["rescues"]) for r in rs])
        fail = np.array([float(r["failed_interventions"]) for r in rs])
        rows.append({
            "problem": prob, "algorithm": alg, "variant": var,
            "mean_activations": act.mean(), "mean_applied": app.mean(),
            "mean_rescues": res.mean(), "mean_failed": fail.mean(),
            "rescue_rate_%": 100.0 * res.sum() / app.sum() if app.sum() > 0 else 0.0,
        })
    path = os.path.join(out, "oracle_dynamics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ---------------------------------------------------------------- 3) TESTS
def build_tests(group, problems, out):
    """Wilcoxon pareado Standard vs cada variante XAI. Reporta identidad."""
    rows = []
    combos = [(p, a) for p in problems for a in ALGS]
    for (prob, alg) in tqdm(combos, desc="Tests Wilcoxon"):
        base, runs_b = series(group, prob, alg, "Standard")
        pvals, labels = [], []
        for var in XAI:
            cur, runs_c = series(group, prob, alg, var)
            mask = np.isfinite(base) & np.isfinite(cur)
            b, c = base[mask], cur[mask]
            diff = c - b
            identical = np.allclose(diff, 0.0, atol=1e-12)
            if identical:
                rows.append({"problem": prob, "algorithm": alg,
                             "comparison": f"Standard vs {var}",
                             "n": int(mask.sum()), "status": "IDÉNTICAS (oráculo no alteró f)",
                             "wilcoxon_p": "", "p_holm": "",
                             "median_diff": 0.0})
                continue
            if HAVE_SCIPY:
                try:
                    stat, p = wilcoxon(b, c)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            pvals.append(p); labels.append(var)
            rows.append({"problem": prob, "algorithm": alg,
                         "comparison": f"Standard vs {var}",
                         "n": int(mask.sum()), "status": "test aplicado",
                         "wilcoxon_p": p, "p_holm": None,
                         "median_diff": float(np.median(diff))})
        # Holm sobre los p reales de este bloque
        if pvals:
            adj = holm(np.array([p if np.isfinite(p) else 1.0 for p in pvals]))
            it = iter(zip(labels, adj))
            la = dict(it)
            for row in rows:
                if (row["problem"] == prob and row["algorithm"] == alg
                        and row["status"] == "test aplicado"):
                    var = row["comparison"].split("vs ")[1]
                    if var in la:
                        row["p_holm"] = float(la[var])
    path = os.path.join(out, "tests_wilcoxon_holm.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "algorithm", "comparison",
                                          "n", "status", "wilcoxon_p", "p_holm",
                                          "median_diff"])
        w.writeheader(); w.writerows(rows)
    return rows


# ------------------------------------------------ 4) CONVERGENCIA (opcional)
def build_convergence(conv_path, out):
    rows = list(csv.DictReader(open(conv_path)))
    # agrega: media de best_fit en bins de NFE por (prob,alg,var)
    agg = defaultdict(lambda: defaultdict(list))
    for r in tqdm(rows, desc="Convergencia"):
        key = (r["problem"], r["algorithm"], r["variant"])
        agg[key][int(float(r["nfe"]))].append(float(r["best_fit"]))
    out_rows = []
    for key, byfe in agg.items():
        for fe in sorted(byfe):
            vals = np.array(byfe[fe])
            out_rows.append({"problem": key[0], "algorithm": key[1],
                             "variant": key[2], "nfe": fe,
                             "mean_best": vals.mean(), "median_best": np.median(vals)})
    path = os.path.join(out, "convergence_mean.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    return out_rows


# ---------------------------------------------------------- TABLA LaTeX
def latex_table(stats_rows, out):
    def fmt(v):
        if v != v:  # nan
            return "--"
        if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.3e}"
        return f"{v:.4f}"
    by = defaultdict(dict)
    for r in stats_rows:
        by[(r["problem"], r["algorithm"])][r["variant"]] = r
    lines = [r"\begin{table*}[t]\centering",
             r"\caption{Estadísticos de $f$ sobre 25 corridas (protocolo CEC2020 RW-CMOP: $D\cdot10^4$ NFE). "
             r"``Feas\%'' es la tasa de factibilidad; ``act'' las activaciones medias del oráculo.}",
             r"\label{tab:rwcmop-stats}", r"\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Problema & Variante & Best & Mediana & Media & Std & Worst & CV med. & Feas\% \\",
             r"\midrule"]
    for (prob, alg) in sorted(by):
        lines.append(r"\multicolumn{9}{l}{\textit{%s -- %s}} \\" % (prob, alg))
        for var in VARIANTS:
            r = by[(prob, alg)].get(var)
            if not r:
                continue
            name = "Estándar" if var == "Standard" else var.replace("-CF", "")
            lines.append(
                f"  {name} & & {fmt(r['best'])} & {fmt(r['median'])} & "
                f"{fmt(r['mean'])} & {fmt(r['std'])} & {fmt(r['worst'])} & "
                f"{fmt(r['cv_mean'])} & {r['feasibility_rate']:.0f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    path = os.path.join(out, "stats_table.tex")
    open(path, "w").write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", default="per_run.csv")
    ap.add_argument("--conv", default=None)
    ap.add_argument("--interv", default=None)
    ap.add_argument("--out", default="analisis_rwcmop")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Cargando per_run...")
    _, group, problems = load(args.per)
    print(f"  {len(problems)} problemas: {problems}")

    stats = build_stats(group, problems, args.out)
    build_oracle_dynamics(group, problems, args.out)
    build_tests(group, problems, args.out)
    latex_table(stats, args.out)

    if args.conv and os.path.exists(args.conv):
        build_convergence(args.conv, args.out)
    else:
        print("  (convergence.csv no provisto: omito curvas de convergencia)")

    print(f"\nListo. Resultados en {args.out}/")
    for fn in sorted(os.listdir(args.out)):
        print("  ", fn)


if __name__ == "__main__":
    main()
