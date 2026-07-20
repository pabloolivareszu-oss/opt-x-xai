#!/usr/bin/env python3
"""
plotter_rwcmop.py  (versión corregida)
======================================
Gráficos del experimento RW-CMOP, coherentes con plotter_eswa.py.

Correcciones respecto a la versión previa:
  1) Convergencia ahora grafica el *gap real* |f_best - f*|. f* se toma de un
     CSV opcional (--fstar-csv problem,f_star) o se infiere como el mínimo
     global por problema sobre todas las variantes/algoritmos/runs.
  2) Se limpian inf/NaN ANTES de np.minimum.accumulate (antes la línea quedaba
     rota en log y el eje GSK aparecía sin ticks).
  3) Se normalizan los nombres de variante en la entrada ("SHAP" -> "SHAP-CF",
     etc.), de modo que el filtro del oráculo y de convergencia no falla en
     silencio cuando el CSV trae el sufijo distinto.
  4) Escala-y: log sólo si el rango efectivo cubre >1 década Y todo > 0; si
     no, linear. Antes se elegía log por "todo positivo" y aplastaba curvas
     casi-constantes contra el piso.
  5) Diagnóstico por (problem, alg, variant) en consola: nº de runs, puntos
     NFE, rango de best_fit, % de no-finitos. Activable con --debug (por
     defecto encendido cuando una figura sale vacía).

Uso:
    python3 plotter_rwcmop.py --conv convergence.csv --per per_run.csv \
        --oracle oracle_dynamics.csv --out figs_rwcmop \
        [--fstar-csv fstar.csv] [--bands] [--debug]
"""
import os, argparse, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- estilo idéntico al experimento principal (plotter_eswa.py) ----
VARIANT_COLORS = {
    "Standard": "tab:blue",
    "SHAP-CF": "tab:orange",
    "LIME-CF": "tab:green",
    "ACME-CF": "tab:red",
    "IBREAKDOWN-CF": "tab:purple",
}
VARIANTS = ["Standard", "SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
XAI = ["SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
ALGS = ["GSK", "SBOA"]
COLOR_AUDIT = "tab:orange"
EPS = 1e-12  # piso para log en gaps muy pequeños

plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def vcolor(v):
    return VARIANT_COLORS.get(str(v), "tab:gray")


# ----------------------------- helpers de limpieza -----------------------
def norm_variant(s):
    """Normaliza nombres de variante: 'SHAP' -> 'SHAP-CF', 'Standard' queda."""
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return s
    if s == "Standard":
        return s
    base = s.replace("-CF", "").replace("_CF", "").strip()
    if base == "Standard":
        return "Standard"
    # admite también minúsculas: 'shap' -> 'SHAP-CF'
    return base.upper() + "-CF" if base.upper() in {"SHAP", "LIME", "ACME", "IBREAKDOWN"} else s


def clean_inputs(conv, per, oracle):
    """Normaliza columnas string y filtra filas obviamente corruptas."""
    for df in (conv, per, oracle):
        if df is None: continue
        if "variant" in df.columns:
            df["variant"] = df["variant"].map(norm_variant)
        for col in ("problem", "algorithm"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    return conv, per, oracle


def ffill_finite(y):
    """Reemplaza inf/NaN por forward-fill; los iniciales no-finitos se rellenan
    con el primer valor finito (backward-fill)."""
    y = np.asarray(y, dtype=float).copy()
    mask = ~np.isfinite(y)
    if not mask.any():
        return y
    last = None
    # forward fill
    for i in range(len(y)):
        if mask[i] and last is not None:
            y[i] = last
        elif not mask[i]:
            last = y[i]
    # backward fill para los iniciales no-finitos
    first_valid = None
    for i in range(len(y)):
        if np.isfinite(y[i]):
            first_valid = y[i]; break
    if first_valid is None:
        return y  # todo era no-finito; lo dejamos para que el caller lo detecte
    for i in range(len(y)):
        if not np.isfinite(y[i]):
            y[i] = first_valid
        else:
            break
    return y


def compute_fstar_auto(conv, val_col):
    """f* por problema = mínimo finito sobre todas las variantes/algs/runs."""
    fstar = {}
    for prob, g in conv.groupby("problem"):
        v = pd.to_numeric(g[val_col], errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        fstar[prob] = float(v.min()) if v.size else 0.0
    return fstar


def load_fstar_csv(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["problem"] = df["problem"].astype(str).str.strip()
    return {r.problem: float(r.f_star) for r in df.itertuples()}


# ----------------------------------------------------- 1) CONVERGENCIA
def plot_convergence(conv, out, is_raw, fstar_map, bands=False, debug=False):
    problems = sorted(conv["problem"].unique())
    val_col = "best_fit" if is_raw else ("median_best" if "median_best" in conv.columns else "mean_best")

    # si no hay mapa externo de f*, inferirlo desde los datos
    if fstar_map is None:
        fstar_map = compute_fstar_auto(conv, val_col)
        print(f"[fstar] inferido automáticamente: {fstar_map}")

    combos = [(p, a) for p in problems for a in ALGS]
    for (prob, alg) in tqdm(combos, desc="Convergencia"):
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        any_plotted = False
        diag_lines = []
        f_star = fstar_map.get(prob, 0.0)
        for var in VARIANTS:
            sub = conv[(conv.problem == prob) & (conv.algorithm == alg)
                       & (conv.variant == var)]
            if sub.empty:
                diag_lines.append(f"    {var}: SIN FILAS")
                continue

            if is_raw:
                grid = np.sort(sub["nfe"].unique())
                if grid.size < 2:
                    diag_lines.append(f"    {var}: sólo {grid.size} punto(s) NFE - sin curva útil")
                    continue
                rows = []
                for r, g in sub.groupby("run"):
                    g = g.sort_values("nfe")
                    rows.append(np.interp(grid, g["nfe"].to_numpy(),
                                          pd.to_numeric(g["best_fit"], errors="coerce").to_numpy()))
                mat = np.vstack(rows)
                # marcar no-finitos como NaN para nanmedian/percentile
                mat = np.where(np.isfinite(mat), mat, np.nan)
                med = np.nanmedian(mat, axis=0)
                q1  = np.nanpercentile(mat, 25, axis=0)
                q3  = np.nanpercentile(mat, 75, axis=0)
                n_runs = mat.shape[0]
                pct_bad = float(np.mean(~np.isfinite(mat))) * 100
            else:
                g = sub.sort_values("nfe")
                grid = g["nfe"].to_numpy()
                med = pd.to_numeric(g[val_col], errors="coerce").to_numpy()
                q1 = q3 = None
                n_runs = np.nan
                pct_bad = float(np.mean(~np.isfinite(med))) * 100

            # limpieza de no-finitos antes del accumulate (BUG #2)
            med = ffill_finite(med)
            if not np.isfinite(med).any():
                diag_lines.append(f"    {var}: todo no-finito - se omite")
                continue

            # gap real (BUG #1)
            gap = np.maximum(med - f_star, EPS)
            gap = np.minimum.accumulate(gap)  # best-so-far monótona

            # eje X normalizado NFE/MaxFEs
            xn = grid / grid.max() if grid.max() > 0 else grid

            ax.plot(xn, gap, lw=2.0, color=vcolor(var), label=var,
                    solid_capstyle="round")
            if bands and q1 is not None and q3 is not None:
                q1g = np.maximum(ffill_finite(q1) - f_star, EPS)
                q3g = np.maximum(ffill_finite(q3) - f_star, EPS)
                q1g = np.minimum.accumulate(q1g)
                q3g = np.minimum.accumulate(q3g)
                ax.fill_between(xn, q1g, q3g, color=vcolor(var), alpha=0.15, lw=0)
            any_plotted = True
            diag_lines.append(
                f"    {var}: {n_runs} runs, {grid.size} pts NFE, "
                f"best ∈[{np.nanmin(med):.4g},{np.nanmax(med):.4g}], "
                f"gap ∈[{gap.min():.3g},{gap.max():.3g}], no-finitos {pct_bad:.1f}%"
            )

        if not any_plotted:
            print(f"[skip] {prob}/{alg}: sin datos graficables")
            for ln in diag_lines: print(ln)
            plt.close(fig); continue

        if debug or not any_plotted:
            print(f"[diag] {prob}/{alg}")
            for ln in diag_lines: print(ln)

        # escala-y robusta (BUG #4): log sólo si rango cubre >1 década y todo>0
        ydata = np.concatenate([l.get_ydata() for l in ax.get_lines()
                                if len(l.get_ydata())])
        ydata = ydata[np.isfinite(ydata)]
        use_log = False
        if ydata.size:
            y_pos = ydata[ydata > 0]
            if y_pos.size and (y_pos.max() / y_pos.min()) > 10.0:
                use_log = True
        ax.set_yscale("log" if use_log else "linear")

        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized function evaluations (NFE / MaxFEs)")
        ax.set_ylabel(r"Best-so-far objective gap $|f_{best}-f^*|$")
        tag = "median, IQR band" if bands else "median, no bands"
        ax.set_title(f"{alg}: Standard and XAI-CF variants ({tag}) · {prob}")
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{prob}_{alg}_Convergence.png"), dpi=300)
        plt.close(fig)


# ----------------------------------------------------- 2) BOXPLOTS
def plot_boxplots(per, out, fstar_map=None, debug=False,
                  strip=True, notched=True, use_gap=True):
    """Boxplot del f final por variante.
    - use_gap=True (default): grafica el gap |f - f*| con f* desde fstar_map
      (o inferido como mínimo por problema). Evita el ofensivo offset '+1'.
    - strip=True: scatter jitter de los runs individuales encima.
    - notched=True: notch en la caja cuando n>=5 (compara medianas a ojo).
    - escala-y: log si el rango positivo > 1 década, lineal si no.
    """
    if per is None or per.empty:
        return

    if use_gap and fstar_map is None:
        fstar_map = {}
        for prob, g in per.groupby("problem"):
            vals = pd.to_numeric(g["f"], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            fstar_map[prob] = float(vals.min()) if vals.size else 0.0
        print(f"[fstar/box] inferido por problema: "
              f"{ {k: f'{v:.6g}' for k,v in fstar_map.items()} }")

    problems = sorted(per["problem"].unique())
    combos = [(p, a) for p in problems for a in ALGS]
    for (prob, alg) in tqdm(combos, desc="Boxplots"):
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        f_star = (fstar_map.get(prob, 0.0) if use_gap else 0.0)
        data, labels, colors, ns = [], [], [], []
        for var in VARIANTS:
            rr = per[(per.problem == prob) & (per.algorithm == alg)
                     & (per.variant == var)]
            vals = pd.to_numeric(rr["f"], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if use_gap:
                vals = np.maximum(vals - f_star, 0.0)
            data.append(vals)
            labels.append(var.replace("-CF", ""))
            colors.append(vcolor(var))
            ns.append(int(vals.size))

        if not data:
            if debug:
                print(f"[skip box] {prob}/{alg}: sin datos finitos")
            plt.close(fig); continue

        positions = np.arange(1, len(data) + 1)
        can_notch = bool(notched) and all(n >= 5 for n in ns)

        bp = ax.boxplot(
            data, positions=positions, widths=0.55,
            tick_labels=labels,
            showmeans=True,
            meanprops=dict(
                marker="^", markerfacecolor="white", markeredgecolor="black",
                markersize=6, markeredgewidth=1.0,
            ),
            showfliers=True,
            flierprops=dict(
                marker="o", markersize=3, markerfacecolor="none",
                markeredgecolor="0.45", markeredgewidth=0.6, alpha=0.55,
            ),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color="0.3", linewidth=1.0),
            capprops=dict(color="0.3", linewidth=1.0),
            boxprops=dict(linewidth=1.0),
            patch_artist=True,
            notch=can_notch,
            bootstrap=2000 if can_notch else None,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)

        if strip:
            rng = np.random.default_rng(42)
            for x, vals, c in zip(positions, data, colors):
                jx = x + rng.uniform(-0.10, 0.10, size=vals.size)
                ax.scatter(jx, vals, s=14, color=c, alpha=0.50,
                           edgecolor="none", zorder=2)

        all_v = np.concatenate(data)
        pos = all_v[all_v > 0]
        use_log = bool(pos.size) and (pos.max() / pos.min() > 10.0)
        if use_log:
            lo = max(np.percentile(pos, 5), pos.min())
            ax.set_yscale("log")
            ax.set_ylim(bottom=lo * 0.5)
            ylab = r"Final objective gap $|f-f^*|$ (log)" if use_gap else r"Final $f$ (log)"
        else:
            ylab = r"Final objective gap $|f-f^*|$" if use_gap else r"Final penalized $f$"
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.set_ylabel(ylab)

        for x, n in zip(positions, ns):
            ax.annotate(f"n={n}", xy=(x, 0), xytext=(0, -26),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="0.4",
                        annotation_clip=False)

        ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
        ax.set_title(f"{prob} — {alg}: distribution of final $f$"
                     + (" (gap to $f^*$)" if use_gap else ""))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25, which="both" if use_log else "major")
        fig.savefig(os.path.join(out, f"{prob}_{alg}_Boxplot.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------- 3) DINÁMICA DEL ORÁCULO
def plot_oracle(oracle, out, debug=False):
    if oracle is None or oracle.empty:
        return
    # tolera columnas alternativas
    col_act = "mean_activations" if "mean_activations" in oracle.columns else \
              ("activations" if "activations" in oracle.columns else None)
    col_res = "mean_rescues" if "mean_rescues" in oracle.columns else \
              ("rescues" if "rescues" in oracle.columns else None)
    if col_act is None or col_res is None:
        print(f"[oracle] columnas no encontradas (esperaba mean_activations/mean_rescues). "
              f"Cols disponibles: {list(oracle.columns)}")
        return

    problems = sorted(oracle["problem"].unique())
    for prob in tqdm(problems, desc="Dinámica oráculo"):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        plotted_any = False
        for ax, alg in zip(axes, ALGS):
            sub = oracle[(oracle.problem == prob) & (oracle.algorithm == alg)
                         & (oracle.variant.isin(XAI))]
            if sub.empty:
                ax.set_title(f"{alg} (sin datos)")
                ax.set_xticks([]); ax.set_yticks([])
                if debug:
                    print(f"[oracle skip] {prob}/{alg}: sub vacío "
                          f"(variantes en CSV: {sorted(oracle.variant.unique())})")
                continue
            x = np.arange(len(XAI)); w = 0.38
            act, res = [], []
            for v in XAI:
                row = sub[sub.variant == v]
                act.append(float(row[col_act].iloc[0]) if not row.empty
                           and np.isfinite(pd.to_numeric(row[col_act], errors="coerce").iloc[0]) else 0.0)
                res.append(float(row[col_res].iloc[0]) if not row.empty
                           and np.isfinite(pd.to_numeric(row[col_res], errors="coerce").iloc[0]) else 0.0)
            if max(act + res) == 0:
                ax.set_title(f"{alg} (todo cero)")
                ax.bar(x - w/2, act, w, label="Activaciones",
                       color="tab:gray", alpha=0.8)
                ax.bar(x + w/2, res, w, label="Rescates",
                       color=COLOR_AUDIT, alpha=0.9)
                ax.set_xticks(x); ax.set_xticklabels(
                    [v.replace("-CF", "") for v in XAI], rotation=20, fontsize=8)
                ax.set_ylim(0, 1)  # evita el -0.05..0.05 del autoscale
                ax.legend(frameon=True, fontsize=8)
                if debug:
                    print(f"[oracle zero] {prob}/{alg}: act={act}, res={res}")
                continue
            ax.bar(x - w/2, act, w, label="Activaciones",
                   color="tab:gray", alpha=0.8)
            ax.bar(x + w/2, res, w, label="Rescates",
                   color=COLOR_AUDIT, alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(
                [v.replace("-CF", "") for v in XAI], rotation=20, fontsize=8)
            ax.set_title(f"{alg}")
            ax.set_ylabel("Media por corrida")
            ax.legend(frameon=True, fontsize=8)
            plotted_any = True
        if not plotted_any:
            print(f"[skip oracle] {prob}: nada para graficar")
            plt.close(fig); continue
        fig.suptitle(f"{prob}: dinámica del oráculo (activaciones vs rescates)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{prob}_OracleDynamics.png"), dpi=300)
        plt.close(fig)


# ----------------------------------------------- diagnóstico previo
def quick_audit(conv, per, oracle):
    print("\n=== AUDITORÍA RÁPIDA DE INSUMOS ===")
    for name, df in [("conv", conv), ("per", per), ("oracle", oracle)]:
        if df is None:
            print(f"  {name}: <ausente>"); continue
        print(f"  {name}: shape={df.shape}, columnas={list(df.columns)}")
        if "variant" in df.columns:
            print(f"    variantes: {sorted(df['variant'].unique())}")
        if "problem" in df.columns:
            print(f"    problems:  {sorted(df['problem'].unique())}")
        if "algorithm" in df.columns:
            print(f"    algs:      {sorted(df['algorithm'].unique())}")
    print("=== FIN AUDITORÍA ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default=None, help="convergence.csv (crudo) o convergence_mean.csv")
    ap.add_argument("--per", default="per_run.csv")
    ap.add_argument("--oracle", default="oracle_dynamics.csv")
    ap.add_argument("--out", default="figs_rwcmop")
    ap.add_argument("--fstar-csv", default=None,
                    help="CSV opcional con columnas problem,f_star para gap real")
    ap.add_argument("--bands", action="store_true",
                    help="añadir banda IQR a la mediana")
    ap.add_argument("--no-strip", action="store_true",
                    help="boxplot: desactiva el jitter de runs individuales")
    ap.add_argument("--no-notch", action="store_true",
                    help="boxplot: desactiva el notch en las cajas")
    ap.add_argument("--box-raw", action="store_true",
                    help="boxplot: usa f crudo en vez del gap (deshabilita el offset '+1')")
    ap.add_argument("--debug", action="store_true",
                    help="imprime diagnóstico por (problema,alg,variante)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conv = pd.read_csv(args.conv) if args.conv and os.path.exists(args.conv) else None
    per  = pd.read_csv(args.per)  if args.per  and os.path.exists(args.per)  else None
    oracle = pd.read_csv(args.oracle) if args.oracle and os.path.exists(args.oracle) else None

    conv, per, oracle = clean_inputs(conv, per, oracle)
    quick_audit(conv, per, oracle)

    fstar_map = load_fstar_csv(args.fstar_csv)

    if conv is not None:
        is_raw = "best_fit" in conv.columns and "run" in conv.columns
        # Si no se pasó fstar externo, lo computamos desde conv para que
        # boxplot y curvas usen el MISMO f* por problema.
        if fstar_map is None:
            val_col = "best_fit" if is_raw else ("median_best"
                       if "median_best" in conv.columns else "mean_best")
            fstar_map = compute_fstar_auto(conv, val_col)
            print(f"[fstar] inferido desde conv: "
                  f"{ {k: f'{v:.6g}' for k,v in fstar_map.items()} }")
        plot_convergence(conv, args.out, is_raw, fstar_map,
                         bands=args.bands, debug=args.debug)
    else:
        print("  (sin archivo de convergencia: omito curvas)")

    plot_boxplots(per, args.out, fstar_map=fstar_map,
                  debug=args.debug,
                  strip=not args.no_strip,
                  notched=not args.no_notch,
                  use_gap=not args.box_raw)
    plot_oracle(oracle, args.out, debug=args.debug)

    print(f"\nListo. Figuras (300 dpi PNG) en {args.out}/")
    for fn in sorted(os.listdir(args.out)):
        print("  ", fn)


if __name__ == "__main__":
    main()
