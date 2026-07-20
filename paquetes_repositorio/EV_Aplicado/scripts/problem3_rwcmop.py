"""
Problem 3: CEC 2020 Real-World Constrained Problems (RW-CMOP) - 10 selected.
----------------------------------------------------------------------------
Each problem exposes the same API:

    info  = PROBLEMS[name]['info']          # dict with D, n_g, n_h, bounds
    f, g, h = PROBLEMS[name]['eval'](x)     # objective, ineqs (<=0), eqs (=0)

We compute the aggregated violation following the standard CEC protocol:

    cv(x) = sum(max(0, g_i))  +  sum(max(0, |h_j| - eps))   (eps = 1e-4)

Problems included (subset of the 57-problem CEC 2020 RW-CMOP suite by Kumar
et al.):
    RW02  Optimal tuned mass damper (3, 3, 0)
    RW08  Process synthesis problem (2, 2, 0)
    RW14  Tension / compression spring design (3, 4, 0)
    RW16  Pressure vessel design (4, 4, 0)
    RW17  Welded beam design (4, 7, 0)
    RW23  Gas transmission compressor design (4, 1, 0)
    RW26  Hydrostatic thrust bearing design (4, 7, 0)
    RW40  Three-bar truss design (2, 3, 0)
    RW41  Himmelblau's nonlinear optimization (5, 6, 0)
    RW45  Speed reducer design (7, 11, 0)

This is a minimal, self-contained reference implementation.
"""

import numpy as np

EPS = 1e-4


# ------------------------------------------------------------------ utilities
def _bounds(lb, ub):
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


def aggregated_violation(g, h, eps=EPS):
    gv = np.sum(np.maximum(0.0, np.asarray(g))) if len(g) else 0.0
    hv = np.sum(np.maximum(0.0, np.abs(np.asarray(h)) - eps)) if len(h) else 0.0
    return float(gv + hv)


# ====================================================================== RW02
# Optimal tuned mass damper (illustrative simplified variant).
def _rw02(x):
    x1, x2, x3 = x
    f = 1.0 + 0.1 * (x1 + x2) ** 2 + (x3 - 1.2) ** 2
    g = [x1 + x2 - 2.0,
         x2 - 2.0 * x1,
         0.5 - x3]
    return f, g, []
RW02 = dict(info=dict(D=3, n_g=3, n_h=0,
                      bounds=_bounds([0, 0, 0.1], [3, 3, 3])),
            eval=_rw02)


# ====================================================================== RW08
# Process synthesis problem (classic benchmark, 2D, nonconvex).
def _rw08(x):
    x1, x2 = x
    f = 2.0 * x1 + x2
    g = [1.25 - x1**2 - x2,
         x1 + x2 - 1.6]
    return f, g, []
RW08 = dict(info=dict(D=2, n_g=2, n_h=0,
                      bounds=_bounds([0, 0], [1.6, 1.0])),
            eval=_rw08)


# ====================================================================== RW14
# Tension / compression spring design (Arora's benchmark).
def _rw14(x):
    x1, x2, x3 = x
    f = (x3 + 2.0) * x2 * x1 ** 2
    g = [1.0 - (x2 ** 3 * x3) / (71785.0 * x1 ** 4),
         (4.0 * x2 ** 2 - x1 * x2) /
         (12566.0 * (x2 * x1 ** 3 - x1 ** 4)) + 1.0 / (5108.0 * x1 ** 2) - 1.0,
         1.0 - (140.45 * x1) / (x2 ** 2 * x3),
         (x1 + x2) / 1.5 - 1.0]
    return f, g, []
RW14 = dict(info=dict(D=3, n_g=4, n_h=0,
                      bounds=_bounds([0.05, 0.25, 2.0], [2.0, 1.3, 15.0])),
            eval=_rw14)


# ====================================================================== RW16
# Pressure vessel design.
def _rw16(x):
    x1, x2, x3, x4 = x
    f = (0.6224 * x1 * x3 * x4
         + 1.7781 * x2 * x3 ** 2
         + 3.1661 * x1 ** 2 * x4
         + 19.84  * x1 ** 2 * x3)
    g = [-x1 + 0.0193 * x3,
         -x2 + 0.00954 * x3,
         -np.pi * x3 ** 2 * x4 - (4.0 / 3.0) * np.pi * x3 ** 3 + 1296000.0,
         x4 - 240.0]
    return f, g, []
RW16 = dict(info=dict(D=4, n_g=4, n_h=0,
                      bounds=_bounds([0, 0, 10, 10], [99, 99, 200, 200])),
            eval=_rw16)


# ====================================================================== RW17
# Welded beam design.
def _rw17(x):
    x1, x2, x3, x4 = x
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    tau_max, sigma_max, delta_max = 13600.0, 30000.0, 0.25
    M = P * (L + x2 / 2.0)
    R = np.sqrt(x2 ** 2 / 4.0 + ((x1 + x3) / 2.0) ** 2)
    J = 2.0 * (np.sqrt(2.0) * x1 * x2 *
               (x2 ** 2 / 12.0 + ((x1 + x3) / 2.0) ** 2))
    tau_p = P / (np.sqrt(2.0) * x1 * x2)
    tau_pp = M * R / J
    tau = np.sqrt(tau_p ** 2
                  + 2.0 * tau_p * tau_pp * x2 / (2.0 * R)
                  + tau_pp ** 2)
    sigma = 6.0 * P * L / (x4 * x3 ** 2)
    delta = 4.0 * P * L ** 3 / (E * x4 * x3 ** 3)
    Pc = (4.013 * E * np.sqrt(x3 ** 2 * x4 ** 6 / 36.0) / L ** 2
          * (1.0 - x3 * np.sqrt(E / (4.0 * G)) / (2.0 * L)))
    f = 1.10471 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
    g = [tau - tau_max,
         sigma - sigma_max,
         x1 - x4,
         0.10471 * x1 ** 2 + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0,
         0.125 - x1,
         delta - delta_max,
         P - Pc]
    return f, g, []
RW17 = dict(info=dict(D=4, n_g=7, n_h=0,
                      bounds=_bounds([0.1, 0.1, 0.1, 0.1], [2, 10, 10, 2])),
            eval=_rw17)


# ====================================================================== RW23
# Gas transmission compressor design (simplified, see Kumar et al.).
def _rw23(x):
    x1, x2, x3, x4 = x
    f = (8.61e5 * np.sqrt(x1) * x2 * x3 ** (-2.0 / 3.0) * (x4 ** 2 - 1) ** -0.5
         + 3.69e4 * x3
         + 7.72e8 / x1 * x4 ** 0.219
         - 765.43e6 / x1)
    g = [x4 * x2 ** -2 + x2 ** -2 - 1.0]
    return f, g, []
RW23 = dict(info=dict(D=4, n_g=1, n_h=0,
                      bounds=_bounds([20, 1, 20, 0.1], [50, 10, 50, 60])),
            eval=_rw23)


# ====================================================================== RW26
# Hydrostatic thrust bearing design.
def _rw26(x):
    R, Ro, mu, Q = x
    gamma, C, n, Ws, Pmax, dTmax, hmin, g_acc = (
        0.0307, 0.5, -3.55, 101000.0, 1000.0, 50.0, 0.001, 386.4)
    P = (np.log10(np.log10(8.122e6 * mu + 0.8)) - C) / n
    delT = 2.0 * (10.0 ** P - 560.0)
    Ef = 9336.0 * Q * gamma * C * delT
    h = (2.0 * np.pi * 750.0 / 60.0) ** 2 * 2.0 * np.pi * mu / Ef * (
        R ** 4 / 4.0 - Ro ** 4 / 4.0)
    Po = (6.0 * mu * Q / (np.pi * h ** 3)) * np.log(R / Ro)
    W = np.pi * Po / 2.0 * (R ** 2 - Ro ** 2) / (np.log(R / Ro) + 1e-5)
    f = (Q * Po / 0.7 + Ef) / 12.0
    g = [Ws - W,
         Po - Pmax,
         delT - dTmax,
         hmin - h,
         Ro - R,
         gamma / (g_acc * Po) * (Q / (2.0 * np.pi * R * h)) - 0.001,
         W / (np.pi * (R ** 2 - Ro ** 2)) - 5000.0]
    return f, g, []
RW26 = dict(info=dict(D=4, n_g=7, n_h=0,
                      bounds=_bounds([1, 1, 1e-6, 1], [16, 16, 16e-6, 16])),
            eval=_rw26)


# ====================================================================== RW40
# Three-bar truss design.
def _rw40(x):
    x1, x2 = x
    L, P, sigma = 100.0, 2.0, 2.0
    f = (2.0 * np.sqrt(2.0) * x1 + x2) * L
    g = [(np.sqrt(2.0) * x1 + x2) / (np.sqrt(2.0) * x1 ** 2 + 2.0 * x1 * x2) * P - sigma,
         x2 / (np.sqrt(2.0) * x1 ** 2 + 2.0 * x1 * x2) * P - sigma,
         1.0 / (np.sqrt(2.0) * x2 + x1) * P - sigma]
    return f, g, []
RW40 = dict(info=dict(D=2, n_g=3, n_h=0,
                      bounds=_bounds([1e-3, 1e-3], [1.0, 1.0])),
            eval=_rw40)


# ====================================================================== RW41
# Himmelblau's nonlinear optimization.
def _rw41(x):
    x1, x2, x3, x4, x5 = x
    f = (5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5
         + 37.293239 * x1 - 40792.141)
    u = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    v = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    w = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    g = [-u, u - 92.0,
         -v + 90.0, v - 110.0,
         -w + 20.0, w - 25.0]
    return f, g, []
RW41 = dict(info=dict(D=5, n_g=6, n_h=0,
                      bounds=_bounds([78, 33, 27, 27, 27],
                                     [102, 45, 45, 45, 45])),
            eval=_rw41)


# ====================================================================== RW45
# Speed reducer design (classic 7-variable problem).
def _rw45(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    f = (0.7854 * x1 * x2 ** 2 *
         (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
         + 7.4777 * (x6 ** 3 + x7 ** 3)
         + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    g = [27.0 / (x1 * x2 ** 2 * x3) - 1.0,
         397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0,
         1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0,
         1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0,
         (np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) /
          (110.0 * x6 ** 3)) - 1.0,
         (np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) /
          (85.0 * x7 ** 3)) - 1.0,
         x2 * x3 / 40.0 - 1.0,
         5.0 * x2 / x1 - 1.0,
         x1 / (12.0 * x2) - 1.0,
         (1.5 * x6 + 1.9) / x4 - 1.0,
         (1.1 * x7 + 1.9) / x5 - 1.0]
    return f, g, []
RW45 = dict(info=dict(D=7, n_g=11, n_h=0,
                      bounds=_bounds([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0],
                                     [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])),
            eval=_rw45)


# ----------------------------------------------------------- problem registry
PROBLEMS = {
    "RW02": RW02, "RW08": RW08, "RW14": RW14, "RW16": RW16, "RW17": RW17,
    "RW23": RW23, "RW26": RW26, "RW40": RW40, "RW41": RW41, "RW45": RW45,
}


def eval_problem(name, x):
    """Return (f, cv) for problem `name` at point x."""
    p = PROBLEMS[name]
    f, g, h = p['eval'](np.asarray(x, dtype=float))
    cv = aggregated_violation(g, h)
    return float(f), float(cv)


# ------------------------------ Demo ----------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2)
    print(f"{'Problem':>7s}  {'D':>3s}  {'n_g':>4s}  {'n_h':>4s}   "
          f"{'f(x)':>14s}   {'CV(x)':>10s}")
    print("-" * 60)
    for name, p in PROBLEMS.items():
        lb, ub = p['info']['bounds']
        x = rng.uniform(lb, ub)
        f, cv = eval_problem(name, x)
        print(f"{name:>7s}  {p['info']['D']:>3d}  {p['info']['n_g']:>4d}  "
              f"{p['info']['n_h']:>4d}   {f:>14.4e}   {cv:>10.4e}")
