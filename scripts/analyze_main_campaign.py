#!/usr/bin/env python3
"""
analyze_main_campaign.py
========================
Tablas agregadas de la CAMPAÑA PRINCIPAL (CEC2022) desde Table_Performance.csv,
con el formato estándar de la literatura de metaheurísticas:

  Indicadores canónicos por celda: Best, Mean, Median, Worst, Std, Q1, Q3
  (los seis típicos + cuartiles). Métrica de error: Final_Err = |f_best - f*|.

Genera (todo desglosado por dimensión D=10 y D=20), con tqdm en cada bucle:
  1) per_function_stats_D{d}.csv  -> estadísticos por algoritmo×variante×función
  2) aggregate_stats_D{d}.csv     -> media de error + RANK MEDIO de Friedman
                                     por algoritmo×variante (sobre las funciones)
  3) friedman_D{d}.csv            -> estadístico de Friedman por algoritmo
  4) wilcoxon_holm_D{d}.csv       -> Standard vs cada variante (Wilcoxon+Holm)
                                     con conteo Win/Tie/Loss por función
  5) success_rate_D{d}.csv        -> tasa de TargetReached por celda
  6) aggregate_stats_D{d}.tex     -> tabla LaTeX lista para el paper

Uso:
    python3 analyze_main_campaign.py --perf Table_Performance.csv --out analisis_principal
"""
import os, csv, argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

try:
    from scipy.stats import wilcoxon, friedmanchisquare, rankdata
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

VARIANTS = ["Standard", "SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
XAI = ["SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        try:
            r["_err"] = float(r["Final_Err"])
        except Exception:
            r["_err"] = np.nan
        r["_tgt"] = str(r.get("TargetReached", "")).strip().lower() in ("true", "1", "yes")
    algos = sorted({r["Algorithm"] for r in rows})
    dims = sorted({r["Dimension"] for r in rows})
    funcs = sorted({int(r["Function"]) for r in rows})
    return rows, algos, dims, funcs


def _csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def cell_errs(rows, d, alg, func, var):
    e = [r["_err"] for r in rows if r["Dimension"] == d and r["Algorithm"] == alg
         and int(r["Function"]) == func and r["Variant"] == var]
    return np.array([x for x in e if np.isfinite(x)])


# ----------------------------------------- 1) estadísticos por función
def per_function(rows, d, algos, funcs, out):
    res = []
    combos = [(a, f, v) for a in algos for f in funcs for v in VARIANTS]
    for (a, f, v) in tqdm(combos, desc=f"Stats por función D{d}"):
        e = cell_errs(rows, d, a, f, v)
        if e.size == 0:
            continue
        res.append({
            "algorithm": a, "function": f, "variant": v,
            "best": np.min(e), "mean": np.mean(e), "median": np.median(e),
            "worst": np.max(e), "std": np.std(e, ddof=1) if e.size > 1 else 0.0,
            "q1": np.percentile(e, 25), "q3": np.percentile(e, 75),
            "n": e.size,
        })
    _csv(os.path.join(out, f"per_function_stats_D{d}.csv"), res, list(res[0].keys()))
    return res


# ------------------------------ 2) agregado: error medio + rank Friedman
def aggregate(rows, d, algos, funcs, out):
    res = []
    for a in tqdm(algos, desc=f"Agregado D{d}"):
        # matriz funciones x variantes con la MEDIANA de error por celda
        med = {v: [] for v in VARIANTS}
        per_func_rank = {v: [] for v in VARIANTS}
        for f in funcs:
            vals = {}
            for v in VARIANTS:
                e = cell_errs(rows, d, a, f, v)
                vals[v] = np.median(e) if e.size else np.nan
                med[v].append(vals[v])
            # rank por función (1 = mejor/menor error) entre variantes
            present = [v for v in VARIANTS if np.isfinite(vals[v])]
            if len(present) >= 2 and HAVE_SCIPY:
                rk = rankdata([vals[v] for v in present])
                for v, r in zip(present, rk):
                    per_func_rank[v].append(r)
        for v in VARIANTS:
            arr = np.array([x for x in med[v] if np.isfinite(x)])
            ranks = per_func_rank[v]
            res.append({
                "dimension": d, "algorithm": a, "variant": v,
                "mean_error": np.mean(arr) if arr.size else np.nan,
                "median_error": np.median(arr) if arr.size else np.nan,
                "std_error": np.std(arr, ddof=1) if arr.size > 1 else 0.0,
                "mean_rank": np.mean(ranks) if ranks else np.nan,
                "n_functions": arr.size,
            })
    _csv(os.path.join(out, f"aggregate_stats_D{d}.csv"), res, list(res[0].keys()))
    return res


# ----------------------------------------- 3) Friedman por algoritmo
def friedman(rows, d, algos, funcs, out):
    res = []
    if not HAVE_SCIPY:
        return res
    for a in tqdm(algos, desc=f"Friedman D{d}"):
        # muestras: por función, la mediana de error de cada variante
        cols = {v: [] for v in VARIANTS}
        for f in funcs:
            row = {}
            ok = True
            for v in VARIANTS:
                e = cell_errs(rows, d, a, f, v)
                if e.size == 0:
                    ok = False; break
                row[v] = np.median(e)
            if ok:
                for v in VARIANTS:
                    cols[v].append(row[v])
        samples = [cols[v] for v in VARIANTS if len(cols[v]) > 0]
        if len(samples) >= 3 and len(samples[0]) >= 3:
            try:
                stat, p = friedmanchisquare(*samples)
            except Exception:
                stat, p = np.nan, np.nan
            res.append({"dimension": d, "algorithm": a,
                        "friedman_stat": stat, "p_value": p,
                        "n_functions": len(samples[0])})
    if res:
        _csv(os.path.join(out, f"friedman_D{d}.csv"), res, list(res[0].keys()))
    return res


def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); prev = 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, (m - rank) * pvals[idx]); adj[idx] = min(1.0, prev)
    return adj


# ------------------------ 4) Wilcoxon Standard vs variante + Win/Tie/Loss
def wilcoxon_tests(rows, d, algos, funcs, out):
    res = []
    for a in tqdm(algos, desc=f"Wilcoxon D{d}"):
        pvals, labels = [], []
        for v in XAI:
            b_all, c_all = [], []
            win = tie = loss = 0
            for f in funcs:
                eb = cell_errs(rows, d, a, f, "Standard")
                ec = cell_errs(rows, d, a, f, v)
                if eb.size == 0 or ec.size == 0:
                    continue
                mb, mc = np.median(eb), np.median(ec)
                b_all.append(mb); c_all.append(mc)
                if mc < mb: win += 1          # variante mejor (menor error)
                elif mc > mb: loss += 1
                else: tie += 1
            entry = {"dimension": d, "algorithm": a, "comparison": f"Standard vs {v}",
                     "n_func": len(b_all), "win": win, "tie": tie, "loss": loss,
                     "wilcoxon_p": "", "p_holm": ""}
            if len(b_all) >= 3 and HAVE_SCIPY and not np.allclose(b_all, c_all):
                try:
                    _, p = wilcoxon(b_all, c_all)
                except Exception:
                    p = np.nan
                entry["wilcoxon_p"] = p
                pvals.append(p); labels.append(v)
            else:
                entry["status"] = "sin variación / n<3"
            res.append(entry)
        if pvals:
            adj = holm(np.array([p if np.isfinite(p) else 1.0 for p in pvals]))
            la = dict(zip(labels, adj))
            for e in res:
                if e["algorithm"] == a and e.get("wilcoxon_p") not in ("", None):
                    v = e["comparison"].split("vs ")[1]
                    if v in la:
                        e["p_holm"] = float(la[v])
    fields = ["dimension", "algorithm", "comparison", "n_func", "win", "tie",
              "loss", "wilcoxon_p", "p_holm", "status"]
    for e in res:
        for k in fields:
            e.setdefault(k, "")
    _csv(os.path.join(out, f"wilcoxon_holm_D{d}.csv"), res, fields)
    return res


# ----------------------------------------- 5) tasa de éxito
def success(rows, d, algos, funcs, out):
    res = []
    for a in tqdm(algos, desc=f"Tasa éxito D{d}"):
        for v in VARIANTS:
            sub = [r for r in rows if r["Dimension"] == d and r["Algorithm"] == a
                   and r["Variant"] == v]
            if not sub:
                continue
            res.append({"dimension": d, "algorithm": a, "variant": v,
                        "success_rate_%": 100.0*sum(r["_tgt"] for r in sub)/len(sub),
                        "n_runs": len(sub)})
    _csv(os.path.join(out, f"success_rate_D{d}.csv"), res, list(res[0].keys()))
    return res


# ----------------------------------------- LaTeX del agregado
def latex_aggregate(agg, d, out):
    def fmt(v):
        if v != v: return "--"
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-3): return f"{v:.3e}"
        return f"{v:.4f}"
    by = defaultdict(dict)
    for r in agg:
        by[r["algorithm"]][r["variant"]] = r
    lines = [r"\begin{table*}[t]\centering",
             r"\caption{Error medio $|f_{best}-f^*|$ y rango medio de Friedman sobre las funciones (D=%s). "
             r"Rango 1 = mejor. ``Estándar'' es la línea base.}" % d,
             r"\label{tab:agg-D%s}" % d, r"\footnotesize\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Algoritmo & Variante & Error medio & Std & Rango medio \\", r"\midrule"]
    for a in sorted(by):
        lines.append(r"\multirow{5}{*}{%s}" % a)
        for v in VARIANTS:
            r = by[a].get(v)
            if not r: continue
            name = "Estándar" if v == "Standard" else v.replace("-CF", "")
            lines.append(f"  & {name} & {fmt(r['mean_error'])} & "
                         f"{fmt(r['std_error'])} & {fmt(r['mean_rank'])} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, f"aggregate_stats_D{d}.tex"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", default="Table_Performance.csv")
    ap.add_argument("--out", default="analisis_principal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Cargando Table_Performance...")
    rows, algos, dims, funcs = load(args.perf)
    print(f"  {len(algos)} algoritmos, dims {dims}, {len(funcs)} funciones, {len(rows)} filas")
    if not HAVE_SCIPY:
        print("  AVISO: scipy no disponible -> sin Friedman/Wilcoxon (instala scipy)")

    for d in dims:
        print(f"\n=== Dimensión D={d} ===")
        per_function(rows, d, algos, funcs, args.out)
        agg = aggregate(rows, d, algos, funcs, args.out)
        friedman(rows, d, algos, funcs, args.out)
        wilcoxon_tests(rows, d, algos, funcs, args.out)
        success(rows, d, algos, funcs, args.out)
        latex_aggregate(agg, d, args.out)

    print(f"\nListo. Tablas en {args.out}/")
    for fn in sorted(os.listdir(args.out)):
        print("  ", fn)


if __name__ == "__main__":
    main()
