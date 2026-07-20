from __future__ import annotations

"""CEC 2022 Single Objective Bound Constrained Optimization protocol helpers.

Primary thesis execution keeps an exact fixed budget so that the FE ledger remains
fully auditable. The first FE at which error <= 1e-8 is additionally recorded as
CompetitionFEterm, allowing competition-style reporting without hiding any FEs.
"""
from pathlib import Path
from typing import Iterable, List
import numpy as np

BENCHMARK = "CEC2022_SO_BCNO"
CEC_FUNCTIONS = list(range(1, 13))
DIMENSIONS = [10, 20]
RUNS = 30
MAX_FES_BY_DIM = {10: 200_000, 20: 1_000_000}
TARGET_ERROR = 1e-8
SEARCH_RANGE = (-100.0, 100.0)

FUNCTION_GROUPS = {
    1: "Basic-Unimodal",
    2: "Basic-Multimodal",
    3: "Basic-Multimodal",
    4: "Basic-Multimodal",
    5: "Basic-Multimodal",
    6: "Hybrid",
    7: "Hybrid",
    8: "Hybrid",
    9: "Composition",
    10: "Composition",
    11: "Composition",
    12: "Composition",
}
TOPOLOGY = {
    1: "Unimodal",
    2: "Multimodal",
    3: "Multimodal",
    4: "Multimodal",
    5: "Multimodal",
    6: "Hybrid",
    7: "Hybrid",
    8: "Hybrid",
    9: "Composition",
    10: "Composition",
    11: "Composition",
    12: "Composition",
}

def max_fes_for_dimension(dim: int) -> int:
    dim = int(dim)
    if dim not in MAX_FES_BY_DIM:
        raise ValueError(f"CEC2022 supports formal dimensions {sorted(MAX_FES_BY_DIM)}, received D={dim}")
    return int(MAX_FES_BY_DIM[dim])

def official_checkpoint_fes(dim: int, max_fes: int | None = None) -> List[int]:
    """16 official CEC2022 checkpoint locations, k = 0..15."""
    dim = int(dim)
    mf = max_fes_for_dimension(dim) if max_fes is None else int(max_fes)
    points = [max(1, min(mf, int(np.floor((dim ** (k / 5.0 - 3.0)) * mf)))) for k in range(16)]
    return points

def official_seed_index(dim: int, function_id: int, run_id: int, runs: int = RUNS) -> int:
    """Return zero-based index corresponding to the technical-report formula."""
    value = (int(dim) // 10 * int(function_id) * int(runs) + int(run_id)) - int(runs)
    return int(value % 1000)

def load_rand_seeds(path: str | Path) -> np.ndarray:
    values = np.loadtxt(Path(path), dtype=np.int64).reshape(-1)
    if len(values) < 1000:
        raise ValueError(f"Expected at least 1000 official seeds, found {len(values)}")
    return values[:1000]

def official_run_seed(rand_seeds: Iterable[int], dim: int, function_id: int, run_id: int, runs: int = RUNS) -> int:
    values = np.asarray(list(rand_seeds), dtype=np.int64).reshape(-1)
    if len(values) < 1000:
        raise ValueError("Official CEC2022 seed vector must contain at least 1000 values")
    return int(values[official_seed_index(dim, function_id, run_id, runs)])
