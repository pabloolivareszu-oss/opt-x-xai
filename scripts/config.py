from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict
from cec2022_protocol import BENCHMARK, CEC_FUNCTIONS, DIMENSIONS, RUNS, MAX_FES_BY_DIM, TARGET_ERROR

ALGORITHMS = ["PSO", "BA", "OPA", "SBOA", "GWO", "L-SHADE", "GSK", "jSO"]
VARIANTS = ["Standard", "SHAP-CF", "LIME-CF", "ACME-CF", "IBREAKDOWN-CF"]
MH_GROUPS = {
    "CLASSICAL_SWARM": ["PSO", "BA", "GWO"],
    "ADAPTIVE_DE": ["L-SHADE", "jSO"],
    "KNOWLEDGE_BASED": ["GSK"],
    "RECENT_BIOINSPIRED": ["OPA", "SBOA"],
}
PROTOCOL_NAME = "V12_CEC2022_ESWA_CORE_SEPARATED_OFFICIAL_EARLYSTOP_GLOBAL_FE_GATE_XAI_CF"

@dataclass(frozen=True)
class OracleConfig:
    trigger_window_fraction: float = 0.02
    history_window_fraction: float = 0.02
    post_window_fraction: float = 0.01
    cooldown_fraction: float = 0.01
    history_snapshots: int = 3
    max_history_buffer_snapshots: int = 64
    probe_iters: int = 1
    probe_population_cap: int = 20
    probe_repeats: int = 1
    shap_queries: int = 16
    lime_queries: int = 12
    acme_grid_points_per_feature: int = 3
    ibreakdown_max_features: int = 6
    ibreakdown_pair_checks: int = 4
    cf_configurations_per_activation: int = 1
    intervention_strength: float = 0.10
    meaningful_optimizer_improvement_rel_threshold: float = 1e-8
    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

DEFAULT_ORACLE_CONFIG = OracleConfig()

def protocol_dict() -> Dict[str, object]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "benchmark": BENCHMARK,
        "cec_functions": CEC_FUNCTIONS,
        "dimensions": DIMENSIONS,
        "runs": RUNS,
        "max_fes_by_dimension": MAX_FES_BY_DIM,
        "target_error": TARGET_ERROR,
        "search_range": [-100.0, 100.0],
        "algorithms": ALGORITHMS,
        "mh_groups": MH_GROUPS,
        "variants": VARIANTS,
        "oracle": DEFAULT_ORACLE_CONFIG.as_dict(),
        "primary_execution_mode": "cec2022_official_early_stop",
        "strict_invariant": "OptimizerFEs + ExplanationFEs + DirectionProbeFEs + CFValidationFEs == TotalFEs == FEterm <= MaxFEs for every completed run",
        "cec2022_reporting": "CompetitionFEterm equals TotalFEs and records the first FE where error < 1e-8, or MaxFEs if the target is not reached. Official 16 checkpoint errors and 17xN matrices are exported during postprocess.",
        "architecture": {
            "oracle": "Stagnation Monitor + FE Availability Gate",
            "xai": "Algorithmic Contribution Estimator",
            "cf": "Deterministic Explanation-Guided Parameter Update",
            "intervention": "Immediate Application + Post-hoc Audit",
        },
        "convergence_axis": "Core stores raw TotalFEs and FEterm. ESWA plotters use normalized FE/MaxFEs by default.",
        "xai_gate": "Global FE availability only: remaining TotalFEs must cover the complete XAI phase. No hard oracle cap.",
        "stagnation_retrigger_policy": "If stagnation persists after cooldown and sufficient global FE budget remains to complete the full XAI phase, XAI may explain and intervene again. There is no separate hard oracle cap. Every activation is ledgered and audited.",
        "postprocess_policy": "Core execution, statistical analysis and ESWA figure generation are separate stages. The core never invokes plotters.",
        "future_work": "CEC2017 replication and cross-benchmark transfer analysis are intentionally deferred to future work.",
    }
