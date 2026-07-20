#!/usr/bin/env python3
"""
run_rwcmop_real.py
==================================================================
Caracterización del comportamiento del framework XAI (motor real de la tesis)
sobre problemas RW-CMOP (CEC 2020 Real-World Constrained, Kumar et al.).

Algoritmos: GSK y SBOA.  Variantes (5): Standard + SHAP/LIME/ACME/IBREAKDOWN-CF.
Problemas:  RW26, RW41.
Protocolo (del PDF de formulación): D*1e4 NFE por corrida, 25 corridas
independientes, semillas pareadas (CRN). Métricas reportadas: best, mean,
median, worst, std de f; violación media (CV); tasa de factibilidad.

NO interpreta resultados: solo ejecuta y vuelca los datos crudos para análisis
posterior (tablas, convergencia, tests Wilcoxon/Friedman-Holm).

Paralelo (joblib) y resumible (cada corrida -> CSV propio en raw/; relanzar SALTA).
"""
import os, json, time, argparse, csv, glob
import numpy as np
np.seterr(all="ignore")          # los problemas RW pueden emitir overflow/invalid
from joblib import Parallel, delayed

from engines import run_engine
from config import DEFAULT_ORACLE_CONFIG
from utils import stable_seed
from ev_problem_rwcmop import RWCMOPProblem
import problem3_rwcmop as P3

ALGORITHMS = ["GSK", "SBOA"]
VARIANTS = ["Standard", "SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
PROBLEMS = ["RW02", "RW08", "RW14", "RW16", "RW17",
            "RW23", "RW26", "RW40", "RW41", "RW45"]
FUNC_ID = {"RW02": 2, "RW08": 8, "RW14": 14, "RW16": 16, "RW17": 17,
           "RW23": 23, "RW26": 26, "RW40": 40, "RW41": 41, "RW45": 45}


def seed_for(dim, func_id, algorithm, run_id):
    """CRN idéntico a la tesis: la variante NO entra en la semilla."""
    return stable_seed("v11_cec2022", dim, func_id, algorithm, run_id)


def downsample(hist, n=140):
    if not hist:
        return []
    a = np.asarray(hist, float)
    if len(a) <= n:
        return [(int(fe), float(v)) for fe, v in a]
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return [(int(a[i, 0]), float(a[i, 1])) for i in idx]


def _csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def task_key(prob, alg, var, run):
    return f"{prob}_{alg}_{var}_run{run}"


def run_one(raw, prob, alg, var, run, max_nfe):
    key = task_key(prob, alg, var, run)
    per_path = os.path.join(raw, f"{key}__perrun.csv")
    if os.path.exists(per_path):
        return f"[SKIP] {key}"
    func = RWCMOPProblem(prob)
    seed = seed_for(func.dim, FUNC_ID[prob], alg, run)
    t0 = time.time()
    res = run_engine(func, func.dim, max_nfe, alg, var, DEFAULT_ORACLE_CONFIG,
                     run_id=run, function_id=FUNC_ID[prob], seed=seed,
                     seed_mode="paired")
    dt = time.time() - t0
    best_x = np.asarray(res["best_x"], float) if res["best_x"] else None
    dec = func.decode(best_x) if best_x is not None else {"f": float("nan"),
                                                          "cv": float("nan"),
                                                          "feasible": 0}
    per = [{
        "problem": prob, "algorithm": alg, "variant": var, "run": run,
        "seed": seed, "final_fit": res["final_fit"],
        "f": dec["f"], "cv": dec["cv"], "feasible": dec["feasible"],
        "oracle_activations": res["oracle_activations"],
        "interventions_applied": res["interventions_applied"],
        "rescues": res["rescues"],
        "failed_interventions": res["failed_interventions"],
        "final_diversity_norm": res.get("final_diversity_norm", ""),
        "oracle_time_ms": round(res["oracle_time_ms"], 2),
        "time_s": round(dt, 3),
        "best_x_json": json.dumps(list(best_x)) if best_x is not None else "",
    }]
    _csv(per_path, per, list(per[0].keys()))

    pts = [(h[0], h[1]) for h in res.get("history", [])
           if isinstance(h, (list, tuple)) and len(h) >= 2]
    conv = [{"problem": prob, "algorithm": alg, "variant": var, "run": run,
             "nfe": fe, "best_fit": v} for fe, v in downsample(pts)]
    _csv(os.path.join(raw, f"{key}__conv.csv"), conv,
         ["problem", "algorithm", "variant", "run", "nfe", "best_fit"])

    # trayectoria del oráculo (param_trace): parámetros, diversidad,
    # estancamiento por generación. Igual que la campaña principal.
    traj = []
    for p in res.get("param_trace", []):
        row = {"problem": prob, "algorithm": alg, "variant": var, "run": run}
        row.update(p)
        traj.append(row)
    if traj:
        _csv(os.path.join(raw, f"{key}__traj.csv"), traj, list(traj[0].keys()))

    interv = [{
        "problem": prob, "algorithm": alg, "variant": var, "run": run,
        "trigger_fe": L.get("Trigger_TotalFEs"), "applied": L.get("Applied"),
        "useful": L.get("Useful"), "realized_gain": L.get("Realized_Intervention_Gain"),
        "status": L.get("Status"),
    } for L in res.get("intervention_logs", [])]
    _csv(os.path.join(raw, f"{key}__interv.csv"), interv,
         ["problem", "algorithm", "variant", "run", "trigger_fe", "applied",
          "useful", "realized_gain", "status"])
    return f"[OK]   {key} f={dec['f']:.4f} cv={dec['cv']:.4e} feas={dec['feasible']} act={res['oracle_activations']}"


def aggregate(out, raw):
    def rd(p):
        with open(p) as f:
            return list(csv.DictReader(f))
    per = [r for p in sorted(glob.glob(os.path.join(raw, "*__perrun.csv"))) for r in rd(p)]
    conv = [r for p in sorted(glob.glob(os.path.join(raw, "*__conv.csv"))) for r in rd(p)]
    interv = [r for p in sorted(glob.glob(os.path.join(raw, "*__interv.csv"))) for r in rd(p)]
    traj = [r for p in sorted(glob.glob(os.path.join(raw, "*__traj.csv"))) for r in rd(p)]
    if per: _csv(os.path.join(out, "per_run.csv"), per, list(per[0].keys()))
    if conv: _csv(os.path.join(out, "convergence.csv"), conv, list(conv[0].keys()))
    if interv: _csv(os.path.join(out, "interventions.csv"), interv, list(interv[0].keys()))
    if traj:
        # GSK y SBOA exponen parámetros distintos -> unión de columnas
        allcols = []
        for r in traj:
            for c in r:
                if c not in allcols:
                    allcols.append(c)
        for r in traj:
            for c in allcols:
                r.setdefault(c, "")
        _csv(os.path.join(out, "trajectory.csv"), traj, allcols)

    # best_solutions: mejor vector por (problema, algoritmo, variante) según Deb
    best = {}
    for r in per:
        k = (r["problem"], r["algorithm"], r["variant"])
        cv = float(r["cv"]) if r["cv"] not in ("", "nan") else 1e18
        ff = float(r["f"]) if r["f"] not in ("", "nan") else 1e18
        cur = (cv, ff)
        def deb(a, b):
            if a[0] <= 1e-9 and b[0] <= 1e-9: return a[1] < b[1]
            if a[0] <= 1e-9: return True
            if b[0] <= 1e-9: return False
            return a[0] < b[0]
        if k not in best or deb(cur, best[k]["key"]):
            best[k] = {"key": cur, "row": r}
    brows = [{"problem": k[0], "algorithm": k[1], "variant": k[2],
              "f": v["row"]["f"], "cv": v["row"]["cv"],
              "feasible": v["row"]["feasible"], "run": v["row"]["run"],
              "best_x_json": v["row"].get("best_x_json", "")}
             for k, v in sorted(best.items())]
    if brows: _csv(os.path.join(out, "best_solutions.csv"), brows, list(brows[0].keys()))

    import statistics as st
    by = {}
    for r in per:
        by.setdefault((r["problem"], r["algorithm"], r["variant"]), []).append(r)
    summ = []
    for (prob, alg, var), rows in sorted(by.items()):
        fs = [float(r["f"]) for r in rows if r["f"] not in ("", "nan") and np.isfinite(float(r["f"]))]
        cvs = [float(r["cv"]) for r in rows if r["cv"] not in ("", "nan")]
        if not fs:
            fs = [float("nan")]
        summ.append({
            "problem": prob, "algorithm": alg, "variant": var,
            "best": min(fs), "median": st.median(fs), "mean": st.mean(fs),
            "std": st.pstdev(fs) if len(fs) > 1 else 0.0, "worst": max(fs),
            "cv_mean": st.mean(cvs) if cvs else float("nan"),
            "feasibility_rate": 100.0*sum(int(r["feasible"]) for r in rows)/len(rows),
            "mean_activations": st.mean([float(r["oracle_activations"]) for r in rows]),
            "mean_rescues": st.mean([float(r["rescues"]) for r in rows]),
        })
    if summ: _csv(os.path.join(out, "summary.csv"), summ, list(summ[0].keys()))
    return len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=str, default="results_rwcmop_real")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3 runs, NFE=5000")
    args = ap.parse_args()

    raw = os.path.join(args.out, "raw"); os.makedirs(raw, exist_ok=True)
    if args.aggregate_only:
        print(f"[aggregate-only] {aggregate(args.out, raw)} corridas en {args.out}/")
        return

    runs = 3 if args.quick else args.runs
    def nfe_for(prob):
        # MaxFEs fijo de 200.000, alineado con la campaña principal (D=10).
        # Las dimensiones RW son nativas (2-7), no se modifican.
        return 5000 if args.quick else 200000

    tasks = [(p, a, v, r) for p in PROBLEMS for a in ALGORITHMS
             for v in VARIANTS for r in range(runs)]
    print("=== RW-CMOP (motor real): GSK & SBOA, 5 variantes, 10 problemas ===")
    print(f"tareas={len(tasks)} runs={runs} n_jobs={args.n_jobs} "
          f"MaxFEs={'5000 (quick)' if args.quick else '200000'} -> {args.out}/")
    t0 = time.time()
    out = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(raw, p, a, v, r, nfe_for(p)) for (p, a, v, r) in tasks)
    for line in out:
        print(line)
    n = aggregate(args.out, raw)
    meta = {
        "benchmark": "CEC2020 RW-CMOP (Kumar et al.)", "problems": PROBLEMS,
        "engine": "thesis engines.run_engine (real oracle + budget + CRN)",
        "algorithms": ALGORITHMS, "variants": VARIANTS, "runs": runs,
        "protocol": "MaxFEs=200000/run (alineado con campaña principal D=10), 50 runs, paired CRN",
        "oracle_config": DEFAULT_ORACLE_CONFIG.as_dict(),
        "total_time_s": round(time.time()-t0, 1), "runs_aggregated": n,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nListo en {meta['total_time_s']}s. {n} corridas en {args.out}/")
    print("  per_run.csv convergence.csv interventions.csv trajectory.csv "
          "best_solutions.csv summary.csv meta.json")


if __name__ == "__main__":
    main()
