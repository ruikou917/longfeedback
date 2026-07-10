"""Credit assignment methods and diagnostics."""

from longfeedback.credit.metrics import (
    credit_recovery_summary,
    kendall_tau,
    normalized_rmse,
    spearman_by_temporal_distance,
)
from longfeedback.credit.oracle import (
    ContinuationMode,
    CounterfactualPair,
    OracleCreditEstimate,
    counterfactual_pair,
    estimate_oracle_credit,
    exact_deterministic_credit,
)

__all__ = [
    "ContinuationMode",
    "CounterfactualPair",
    "OracleCreditEstimate",
    "counterfactual_pair",
    "credit_recovery_summary",
    "estimate_oracle_credit",
    "exact_deterministic_credit",
    "kendall_tau",
    "normalized_rmse",
    "spearman_by_temporal_distance",
]
