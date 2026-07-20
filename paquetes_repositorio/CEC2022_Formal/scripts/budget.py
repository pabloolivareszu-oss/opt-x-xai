from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

class BudgetExhausted(RuntimeError):
    """Raised when a real objective evaluation cannot be paid from MaxFEs."""

class TargetReached(RuntimeError):
    """Raised when a new objective evaluation is requested after the CEC2022 target was reached."""

@dataclass
class BudgetSnapshot:
    max_fes: int
    optimizer_fes: int
    explanation_fes: int
    direction_probe_fes: int
    cf_validation_fes: int
    total_fes: int
    remaining_fes: int
    oracle_fes: int
    budget_valid: bool
    budget_exhausted: bool
    last_improvement_fe: int
    last_optimizer_improvement_fe: int
    last_meaningful_optimizer_improvement_fe: int
    best_optimizer_fitness: float
    last_meaningful_optimizer_fitness: float
    target_error: float
    target_reached: bool
    first_target_fe: int
    competition_feterm: int

class StrictEvaluationBudget:
    """Single mandatory ledger for every objective-function evaluation.

    A CEC2022 run is formally complete when it reaches the target error or exhausts MaxFEs.
    CompetitionFEterm is the true termination FE and all FE categories must sum exactly to TotalFEs.
    """
    VALID_CATEGORIES = {"optimizer", "explanation", "direction_probe", "cf_validation"}

    def __init__(self, objective: Callable[[np.ndarray], float], max_fes: int, optimum: float,
                 *, meaningful_optimizer_improvement_rel_threshold: float = 1e-8,
                 target_error: float = 1e-8,
                 on_improvement: Optional[Callable[[int, float, np.ndarray, str], None]] = None) -> None:
        self._objective = objective
        self.max_fes = int(max_fes)
        self.optimum = float(optimum)
        self.optimizer_fes = 0
        self.explanation_fes = 0
        self.direction_probe_fes = 0
        self.cf_validation_fes = 0
        self.best_fitness = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.last_improvement_fe = 0
        self.last_optimizer_improvement_fe = 0
        self.last_meaningful_optimizer_improvement_fe = 0
        self.best_optimizer_fitness = float("inf")
        self.last_meaningful_optimizer_fitness = float("inf")
        self.meaningful_optimizer_improvement_rel_threshold = float(meaningful_optimizer_improvement_rel_threshold)
        self.target_error = float(target_error)
        self.first_target_fe = 0
        self.history: List[Tuple[int, float, str]] = []
        self._on_improvement = on_improvement

    @property
    def oracle_fes(self) -> int:
        return int(self.explanation_fes + self.direction_probe_fes + self.cf_validation_fes)

    @property
    def total_fes(self) -> int:
        return int(self.optimizer_fes + self.explanation_fes + self.direction_probe_fes + self.cf_validation_fes)

    @property
    def remaining_fes(self) -> int:
        return int(self.max_fes - self.total_fes)

    @property
    def exhausted(self) -> bool:
        return self.total_fes >= self.max_fes

    def can_reserve(self, n: int) -> bool:
        return int(n) >= 0 and self.remaining_fes >= int(n)

    def _charge(self, category: str, n: int = 1) -> None:
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Unknown FE category: {category}")
        if not self.can_reserve(n):
            raise BudgetExhausted(f"Cannot reserve {n} FEs for {category}: remaining={self.remaining_fes}, max={self.max_fes}")
        setattr(self, f"{category}_fes", int(getattr(self, f"{category}_fes")) + int(n))

    def _observe(self, arr: np.ndarray, value: float, category: str, fe_index: int) -> None:
        # Optimizer-level progress is tracked independently from the global incumbent.
        # This matters when XAI probes find a solution that is globally better: the
        # baseline optimizer can still recover later without necessarily beating the
        # probe incumbent immediately.
        if category == "optimizer" and value < self.best_optimizer_fitness:
            self.best_optimizer_fitness = float(value)
            self.last_optimizer_improvement_fe = int(fe_index)
            ref = self.last_meaningful_optimizer_fitness
            meaningful = not np.isfinite(ref)
            if np.isfinite(ref):
                gain = float(ref - value)
                relative_gain = gain / max(abs(float(ref)), 1.0)
                meaningful = relative_gain >= self.meaningful_optimizer_improvement_rel_threshold
            if meaningful:
                self.last_meaningful_optimizer_fitness = float(value)
                self.last_meaningful_optimizer_improvement_fe = int(fe_index)

        if value < self.best_fitness:
            self.best_fitness = float(value)
            self.best_x = arr.copy()
            self.last_improvement_fe = int(fe_index)
            self.history.append((int(fe_index), float(value), str(category)))
            if self._on_improvement is not None:
                self._on_improvement(int(fe_index), float(value), arr.copy(), str(category))
        if self.first_target_fe == 0 and max(0.0, float(self.best_fitness) - float(self.optimum)) < self.target_error:
            self.first_target_fe = int(fe_index)

    def evaluate(self, x: Sequence[float], *, category: str) -> float:
        if self.first_target_fe > 0:
            raise TargetReached(f"CEC2022 target already reached at FE={self.first_target_fe}")
        self._charge(category, 1)
        arr = np.asarray(x, dtype=float)
        value = float(self._objective(arr))
        self._observe(arr, value, category, self.total_fes)
        return value

    def evaluate_many(self, xs: Iterable[Sequence[float]], *, category: str) -> np.ndarray:
        values = []
        for x in xs:
            if self.first_target_fe > 0:
                break
            values.append(self.evaluate(x, category=category))
        return np.asarray(values, dtype=float)

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_fes=self.max_fes,
            optimizer_fes=self.optimizer_fes,
            explanation_fes=self.explanation_fes,
            direction_probe_fes=self.direction_probe_fes,
            cf_validation_fes=self.cf_validation_fes,
            total_fes=self.total_fes,
            remaining_fes=self.remaining_fes,
            oracle_fes=self.oracle_fes,
            budget_valid=self.total_fes <= self.max_fes,
            budget_exhausted=self.exhausted,
            last_improvement_fe=self.last_improvement_fe,
            last_optimizer_improvement_fe=self.last_optimizer_improvement_fe,
            last_meaningful_optimizer_improvement_fe=self.last_meaningful_optimizer_improvement_fe,
            best_optimizer_fitness=self.best_optimizer_fitness,
            last_meaningful_optimizer_fitness=self.last_meaningful_optimizer_fitness,
            target_error=self.target_error,
            target_reached=bool(self.first_target_fe > 0),
            first_target_fe=int(self.first_target_fe),
            competition_feterm=int(self.first_target_fe if self.first_target_fe > 0 else self.max_fes),
        )

    def assert_valid(self, *, require_exhausted: bool = False) -> None:
        expected = self.optimizer_fes + self.explanation_fes + self.direction_probe_fes + self.cf_validation_fes
        if self.total_fes != expected:
            raise AssertionError("Budget categories do not sum to TotalFEs")
        if self.total_fes > self.max_fes:
            raise AssertionError(f"Budget exceeded: total={self.total_fes}, max={self.max_fes}")
        if require_exhausted and self.total_fes != self.max_fes:
            raise AssertionError(f"Incomplete run: total={self.total_fes}, expected exact MaxFEs={self.max_fes}")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self.snapshot())
