from __future__ import annotations

"""Incremental trajectory writer.

Selected runs store one compressed population snapshot per optimizer iteration.
Nothing is accumulated in RAM. A manifest links snapshots to TotalFEs.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import csv
import numpy as np

@dataclass
class TrajectoryRow:
    Generation: int
    TotalFEs: int
    OptimizerFEs: int
    ExplanationFEs: int
    DirectionProbeFEs: int
    CFValidationFEs: int
    PopulationSize: int
    File: str

class IncrementalTrajectoryWriter:
    def __init__(self, directory: Path | str | None):
        self.directory = Path(directory) if directory is not None else None
        self.count = 0
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.manifest = self.directory / "manifest.csv"
            if not self.manifest.exists():
                with self.manifest.open("w", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=list(TrajectoryRow.__annotations__.keys())).writeheader()

    def write(self, population: np.ndarray, generation: int, budget) -> None:
        if self.directory is None:
            return
        self.count += 1
        filename = f"iter_{int(generation):07d}_fe_{int(budget.total_fes):09d}.npz"
        np.savez_compressed(self.directory / filename, population=np.asarray(population, dtype=float))
        row = TrajectoryRow(
            Generation=int(generation), TotalFEs=int(budget.total_fes), OptimizerFEs=int(budget.optimizer_fes),
            ExplanationFEs=int(budget.explanation_fes), DirectionProbeFEs=int(budget.direction_probe_fes),
            CFValidationFEs=int(budget.cf_validation_fes), PopulationSize=int(len(population)), File=filename,
        )
        with self.manifest.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=list(TrajectoryRow.__annotations__.keys())).writerow(asdict(row))
