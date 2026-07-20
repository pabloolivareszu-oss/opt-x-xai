from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

EPS = 1e-12


def stable_seed(*parts: object, modulo: int = 2**32 - 1) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % modulo


def safe_error(fitness: float, optimum: float) -> float:
    return float(max(0.0, float(fitness) - float(optimum)))


def population_diversity(population: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[float, float]:
    pop = np.asarray(population, dtype=float)
    if pop.size == 0:
        return 0.0, 0.0
    raw = float(np.mean(np.var(pop, axis=0)))
    scale = np.maximum(np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float), EPS)
    normed = (pop - np.asarray(lb, dtype=float)) / scale
    normalized = float(np.mean(np.var(normed, axis=0)))
    return raw, normalized


def relative_improvement(values: Sequence[float], k: int = 10) -> float:
    if len(values) < 2:
        return 1.0
    arr = np.asarray(values[-k:], dtype=float)
    return float(max(0.0, arr[0] - arr[-1]) / max(abs(arr[0]), EPS))


def slope(values: Sequence[float], k: int = 5) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values[-k:], dtype=float)
    return float((arr[-1] - arr[0]) / max(1, len(arr) - 1))


def json_safe(value: Any) -> str:
    def _convert(v: Any):
        if is_dataclass(v):
            return asdict(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): _convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        return v
    return json.dumps(_convert(value), sort_keys=True)


def machine_metadata() -> Dict[str, str]:
    return {
        "Machine_ID": socket.gethostname(),
        "Architecture": platform.machine(),
        "Platform": platform.platform(),
        "Python_Version": sys.version.replace("\n", " "),
        "Processor": platform.processor(),
        "NumPy_Version": np.__version__,
    }


def carry_forward_history(history: Sequence[Tuple[int, float, str]], max_fes: int) -> List[Tuple[int, float, str]]:
    out = list(history)
    if not out:
        return out
    if out[-1][0] < max_fes:
        out.append((int(max_fes), float(out[-1][1]), "carry_forward"))
    return out


def terminalize_history(history: Sequence[Tuple[int, float, str]], terminal_fe: int, final_fitness: float) -> List[Tuple[int, float, str]]:
    """Close a convergence trace at FEterm without inventing paid evaluations.

    The synthetic terminal row is explicitly labelled and is used only to draw
    the final plateau. It never changes the FE ledger.
    """
    out = list(history)
    term = int(terminal_fe)
    if term < 0:
        raise ValueError("terminal_fe must be non-negative")
    if not out:
        return [(term, float(final_fitness), "terminal_carry_forward")]
    if int(out[-1][0]) < term:
        out.append((term, float(out[-1][1]), "terminal_carry_forward"))
    return out
