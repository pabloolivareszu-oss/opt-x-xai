"""Adaptador de los problemas RW-CMOP (CEC2020 real-world) al motor real.

Sanea valores no finitos (algunos problemas como RW23/RW26 producen NaN/inf
en regiones inválidas del espacio): se mapean a una penalización grande pero
finita, de modo que el optimizador los trate como muy infactibles. Práctica
estándar en optimización con restricciones de caja real.
"""
import numpy as np
import problem3_rwcmop as P3

RHO = 1000.0
BIG = 1e12

class RWCMOPProblem:
    def __init__(self, rw_name):
        self.rw_name = rw_name
        info = P3.PROBLEMS[rw_name]['info']
        lb, ub = info['bounds']
        self.lb = np.asarray(lb, float); self.ub = np.asarray(ub, float)
        self.dim = info['D']; self.f_global = 0.0
        self.name = f"RWCMOP_{rw_name}"

    def _safe(self, x):
        try:
            f, cv = P3.eval_problem(self.rw_name, np.asarray(x, float))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return BIG, BIG
        if not np.isfinite(f): f = BIG
        if not np.isfinite(cv): cv = BIG
        return float(f), float(cv)

    def evaluate(self, x):
        f, cv = self._safe(x)
        return float(min(BIG, f + RHO * cv)) if cv > 0 else float(f)

    def decode(self, x):
        f, cv = self._safe(x)
        return {"f": f, "cv": cv, "feasible": int(cv <= 1e-9)}
