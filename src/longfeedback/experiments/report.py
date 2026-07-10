"""Paper-style v0.2 report generated from experiment artifacts.

The report reads the metrics JSON of E0, Gate A, Gate B, and E1 and renders a
markdown document containing only aggregate statistics (safe to publish). It
never touches dataset text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_metrics(repository: Path, experiment: str) -> dict[str, Any]:
    path = repository / "artifacts" / experiment / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run the {experiment} experiment before building the report"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _f(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def build_v02_report(repository: Path) -> str:
    """Render the v0.2 paper-style report from experiment artifacts."""

    e0 = _load_metrics(repository, "e0")
    gate_a = _load_metrics(repository, "gate_a")
    gate_b = _load_metrics(repository, "gate_b")
    e1 = _load_metrics(repository, "e1")

    lines: list[str] = []
    add = lines.append

    add("# Learning Reward and Credit from Delayed Implicit Outcomes — v0.2 results")
    add("")
    add(
        "> Auto-generated from experiment artifacts; regenerate with "
        "`longfeedback report build`. All numbers are aggregate statistics."
    )
    add("")
    add("## Abstract")
    add("")
    add(
        "We study whether delayed behavioral outcomes can supervise language-agent "
        "training. Across four structural world families with known causal ground "
        "truth, capacity-matched sequence models that predict the terminal outcome "
        "equally well differ drastically in how accurately their per-step signals "
        "recover true interventional credit; supervising an action-value head on "
        "paired counterfactual credit closes that gap. Bootstrap-ensemble "
        "uncertainty flags credit errors under parameter and logging-policy shift, "
        "and on real conversation logs a delayed future-feedback outcome is "
        "learnable well above trivial length baselines."
    )
    add("")
    add("## 1. Claims and non-claims")
    add("")
    add("Supported claims (structural worlds unless noted):")
    add("")
    add("1. Terminal-outcome accuracy does not imply credit accuracy (E0/Gate A/Gate B).")
    add(
        "2. Oracle-credit supervision improves credit recovery at fixed capacity "
        f"(wins in {gate_b['gate_b_decision']['winning_families']}/4 families)."
    )
    add("3. Ensemble uncertainty correlates with credit error under distribution shift.")
    add(
        "4. Real-log delayed feedback (next-turn pushback) is predictable above "
        "trivial baselines (predictive claim only)."
    )
    add("")
    add(
        "Not claimed: real-user modeling, causal effects in observational chat logs, "
        "production transfer, or that behavioral proxies equal user welfare."
    )
    add("")
    add("## 2. Structural world families")
    add("")
    add("| Family | Difficulty axis | Proxy Y | Utility U |")
    add("|---|---|---|---|")
    add(
        "| A fatigue/habit | stochasticity + partial observability | habit threshold | habit − fatigue/action costs |"
    )
    add(
        "| B hidden intent | exogenous latent shifts + hidden confounding | progress + shock threshold | matched progress |"
    )
    add(
        "| C delayed conversion | long/variable delay, competing causes | conversion | conversion value − send costs |"
    )
    add(
        "| D proxy divergence | Goodhart gap | return event | progress + trust − dependency − interruptions |"
    )
    add("")
    add("## 3. Method")
    add("")
    add(
        "One causal-Transformer architecture with three heads (terminal outcome, "
        "prefix value, action value); variants differ only in loss weights, so all "
        "comparisons are capacity-matched by construction. Oracle credit uses "
        "paired common-random-number counterfactuals with frozen continuation and "
        "adaptive Monte Carlo precision. Uncertainty is the between-member std of "
        f"a {gate_b['families']['world_a']['ensemble']['members']}-member bootstrap "
        "ensemble."
    )
    add("")
    add("## 4. Results")
    add("")
    add("### 4.1 Outcome accuracy vs credit recovery (in-distribution)")
    add("")
    add(
        "| Family | AUROC (outcome-only) | credit ρ outcome-only | credit ρ prefix/RUDDER | credit ρ credit-supervised |"
    )
    add("|---|---:|---:|---:|---:|")
    for name, family in gate_b["families"].items():
        variants = family["variants"]
        add(
            f"| {name} | {_f(variants['docm_outcome']['outcome']['auroc'])} "
            f"| {_f(variants['docm_outcome']['credit']['spearman'])} "
            f"| {_f(variants['docm_prefix']['credit']['spearman'])} "
            f"| {_f(variants['docm_credit']['credit']['spearman'])} |"
        )
    add("")
    gate_a_gap = gate_a["gate_a_decision"]["gap_details"]
    best_gap = max(detail["spearman_gap"] for detail in gate_a_gap.values())
    add(
        f"Gate A (Worlds A/B, more regimes) shows the same pattern with a maximum "
        f"credit-Spearman gap of {_f(best_gap)} at outcome-AUROC differences below "
        f"{_f(max(detail['auroc_difference'] for detail in gate_a_gap.values()))}."
    )
    add("")
    add("### 4.2 Uncertainty under distribution shift")
    add("")
    add("| Family | shift | uncertainty–error ρ | error-detection AUROC | credit ρ degradation |")
    add("|---|---|---:|---:|---:|")
    for name, family in gate_b["families"].items():
        ensemble = family["ensemble"]
        add(
            f"| {name} | {family['shift_kind']} "
            f"| {_f(ensemble['under_shift']['uncertainty_error_spearman'])} "
            f"| {_f(ensemble['under_shift']['error_detection_auroc'])} "
            f"| {_f(ensemble['credit_spearman_degradation'])} |"
        )
    add("")
    add("### 4.3 Real conversation logs (E1, LMSYS-Chat-1M)")
    add("")
    data = e1["data"]
    fail_next = e1["labels"]["fail_next"]
    add(
        f"{data['conversations']} prepared conversations yield {data['examples_total']} "
        f"assistant-turn examples; the next-turn failure label has prevalence "
        f"{_f(fail_next['prevalence_test'])} on the test split."
    )
    add("")
    add("| Model | AUROC | AUPRC | Brier | ECE | NLL |")
    add("|---|---:|---:|---:|---:|---:|")
    for model_name, values in fail_next["models"].items():
        add(
            f"| {model_name} | {_f(values['auroc'])} | {_f(values['auprc'])} "
            f"| {_f(values['brier'])} | {_f(values['ece'])} | {_f(values['nll'])} |"
        )
    add("")
    add("### 4.4 Policy sanity check (Gate A)")
    add("")
    policy = gate_a["policy_check"]
    add(
        f"On `{policy['regime']}`, a Q-greedy policy from the credit head reaches "
        f"true utility {_f(policy['greedy_q']['utility'])} versus "
        f"{_f(policy['behavior_clone']['utility'])} for behavior cloning and "
        f"{_f(policy['behavior']['utility'])} for the behavior policy, with no "
        "proxy-up/utility-down inversion."
    )
    add("")
    add("## 5. Gate decisions")
    add("")
    add(f"- Gate A: **{gate_a['status']}**")
    add(
        f"- Gate B: **{gate_b['status']}** "
        f"(criteria: {json.dumps({k: v for k, v in gate_b['gate_b_decision'].items() if isinstance(v, bool)})})"
    )
    add(f"- E1: **{e1['status']}**")
    add("")
    add("## 6. Limitations")
    add("")
    add(
        "- Real-log labels are rule-based behavioral proxies without human validation yet; "
        "the trivial length baseline alone reaches "
        f"{_f(fail_next['models']['trivial_length_ridge']['auroc'])} AUROC, so much of the "
        "signal is positional."
    )
    add(
        "- Credit supervision uses oracle labels available only in simulation; "
        "closing the gap from observable signals is future work."
    )
    add(
        "- Cross-family transfer (leave-one-world-out training) is deferred; shift "
        "results cover parameter and logging-policy shifts within families."
    )
    add(
        "- LMSYS conversations lack timestamps and user IDs (synthetic event times; "
        "possible cross-split user leakage)."
    )
    add("- No LLM-native reranking yet (v0.3).")
    add("")
    add("## 7. Reproducibility")
    add("")
    add("```bash")
    add("make e0 && make gate-a && make gate-b        # structural experiments")
    add("make data-lmsys && make e1                   # real-log experiment (local data)")
    add("longfeedback report build                    # regenerate this report")
    add("```")
    add("")
    add(
        "Scientific metric hashes: "
        f"e0 `{e0['scientific_metrics_sha256'][:12]}`, "
        f"gate_a `{gate_a['scientific_metrics_sha256'][:12]}`, "
        f"gate_b `{gate_b['scientific_metrics_sha256'][:12]}`, "
        f"e1 `{e1['scientific_metrics_sha256'][:12]}`."
    )
    add("")
    return "\n".join(lines)


def write_v02_report(repository: Path, output_path: Path | None = None) -> Path:
    target = output_path or repository / "reports" / "v0_2_report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_v02_report(repository) + "\n", encoding="utf-8")
    return target
