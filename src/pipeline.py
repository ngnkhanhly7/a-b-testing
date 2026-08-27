"""
End-to-end pipeline: assignment SRM check -> primary metric test -> sample
size adequacy check -> guardrail metric checks.

This is the module the API layer (src/api/main.py) calls into.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.assignment import SRMResult, check_srm
from src.power import SampleSizeCheck, check_sample_size, required_sample_size
from src.sequential import obrien_fleming_boundary, sequential_z_stat
from src.stats_engine import TestResult, run_test


@dataclass
class GuardrailConfig:
    name: str
    column: str
    metric_type: str  # "proportion" or "continuous"
    lower_is_worse: bool = True  # if True, treatment scoring lower than control is a regression


@dataclass
class ExperimentConfig:
    experiment_id: str
    primary_metric_column: str
    metric_type: str  # "proportion" or "continuous"
    alpha: float = 0.05
    expected_split: dict = field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    baseline_rate: float | None = None  # required for the sample-size adequacy check
    minimum_detectable_effect: float | None = None
    guardrails: list[GuardrailConfig] = field(default_factory=list)
    # Sequential testing (checkpoint 4): if set, every call to analyze_experiment
    # must be given the current look_number, and the primary-metric verdict is
    # gated behind an O'Brien-Fleming boundary instead of the static z-test
    # threshold -- this is what makes repeated peeking during a live experiment
    # safe. Only applies to metric_type="proportion".
    max_looks: int | None = None


@dataclass
class SequentialCheck:
    look_number: int
    max_looks: int
    z_stat: float
    boundary: float
    recommendation: str  # "stop_significant" | "continue" | "stop_no_effect"

    @property
    def message(self) -> str:
        if self.recommendation == "stop_significant":
            return (
                f"Sequential check (look {self.look_number}/{self.max_looks}): "
                f"|z|={abs(self.z_stat):.3f} vượt ngưỡng O'Brien-Fleming {self.boundary:.3f} "
                f"-- CÓ THỂ DỪNG, kết quả có ý nghĩa thống kê và an toàn với việc đã peek nhiều lần."
            )
        if self.recommendation == "stop_no_effect":
            return (
                f"Sequential check (look {self.look_number}/{self.max_looks}): "
                f"đã tới look cuối cùng mà chưa vượt ngưỡng -- DỪNG, kết luận không đủ bằng chứng có khác biệt."
            )
        return (
            f"Sequential check (look {self.look_number}/{self.max_looks}): "
            f"|z|={abs(self.z_stat):.3f} chưa vượt ngưỡng O'Brien-Fleming {self.boundary:.3f} "
            f"-- TIẾP TỤC thu thập thêm dữ liệu, chưa được kết luận dựa trên p-value tĩnh."
        )


@dataclass
class PipelineResult:
    srm: SRMResult
    primary: TestResult
    sample_size_check: SampleSizeCheck | None
    guardrails: dict[str, dict]
    sequential: SequentialCheck | None = None

    def summary(self) -> str:
        lines = []
        if self.srm.is_mismatched:
            lines.append(f"⚠ SRM MISMATCH detected: {self.srm.observed} vs expected {self.srm.expected} (p={self.srm.p_value:.4g}). Kết quả bên dưới KHÔNG đáng tin cậy cho tới khi khắc phục lỗi chia nhóm.")
        else:
            lines.append(f"SRM check OK (p={self.srm.p_value:.4g}).")

        lines.append(f"Primary metric: {self.primary.conclusion}")

        if self.sequential is not None:
            lines.append(self.sequential.message)

        if self.sample_size_check is not None:
            lines.append(self.sample_size_check.message)

        for name, g in self.guardrails.items():
            flag = "⚠ REGRESSION" if g["regression_detected"] else "OK"
            lines.append(f"Guardrail '{name}': {flag} -- {g['test'].conclusion}")

        return "\n".join(lines)


def _test_data_for(df: pd.DataFrame, column: str, metric_type: str) -> dict:
    control = df[df.group == "A"]
    treatment = df[df.group == "B"]
    if metric_type == "proportion":
        return dict(
            conversions_control=int(control[column].sum()),
            n_control=len(control),
            conversions_treatment=int(treatment[column].sum()),
            n_treatment=len(treatment),
        )
    return dict(
        values_control=control[column].to_numpy(dtype=float),
        values_treatment=treatment[column].to_numpy(dtype=float),
    )


def analyze_experiment(user_df: pd.DataFrame, config: ExperimentConfig, look_number: int | None = None) -> PipelineResult:
    """`user_df` must be one row per user with columns: group, plus the
    primary metric column and any guardrail columns.

    `look_number` is which peek at the data this is (1st time checking, 2nd
    time, ...). Pass it whenever `config.max_looks` is set so the primary
    metric verdict is gated by the O'Brien-Fleming boundary rather than the
    static p<alpha threshold -- required to safely check results more than
    once while an experiment is still running (see checkpoint 4).
    """
    group_counts = user_df.group.value_counts().to_dict()
    srm = check_srm(group_counts, config.expected_split)

    primary = run_test(
        _test_data_for(user_df, config.primary_metric_column, config.metric_type),
        config.metric_type,
        alpha=config.alpha,
    )

    sequential = None
    if config.max_looks is not None and look_number is not None and config.metric_type == "proportion":
        control = user_df[user_df.group == "A"]
        treatment = user_df[user_df.group == "B"]
        z_stat = sequential_z_stat(
            conversions_control=int(control[config.primary_metric_column].sum()),
            n_control=len(control),
            conversions_treatment=int(treatment[config.primary_metric_column].sum()),
            n_treatment=len(treatment),
        )
        capped_look = min(look_number, config.max_looks)
        boundary = obrien_fleming_boundary(capped_look, config.max_looks, config.alpha)

        if abs(z_stat) > boundary:
            recommendation = "stop_significant"
        elif look_number >= config.max_looks:
            recommendation = "stop_no_effect"
        else:
            recommendation = "continue"

        sequential = SequentialCheck(
            look_number=look_number,
            max_looks=config.max_looks,
            z_stat=z_stat,
            boundary=boundary,
            recommendation=recommendation,
        )

    sample_size_check = None
    if config.baseline_rate is not None and config.minimum_detectable_effect is not None:
        n_required = required_sample_size(config.baseline_rate, config.minimum_detectable_effect, alpha=config.alpha)
        sample_size_check = check_sample_size(len(user_df), n_required * 2)

    guardrail_results = {}
    for g in config.guardrails:
        gtest = run_test(_test_data_for(user_df, g.column, g.metric_type), g.metric_type, alpha=config.alpha)
        regression = gtest.is_significant and (
            (g.lower_is_worse and gtest.diff < 0) or (not g.lower_is_worse and gtest.diff > 0)
        )
        guardrail_results[g.name] = {"test": gtest, "regression_detected": regression}

    return PipelineResult(
        srm=srm,
        primary=primary,
        sample_size_check=sample_size_check,
        guardrails=guardrail_results,
        sequential=sequential,
    )


def main() -> None:
    """Smoke test: full flow (assign via consistent hashing -> simulate outcomes
    -> aggregate -> analyze, with sequential peeking) on scenario1 (clear
    difference), plus a synthetic guardrail-regression scenario (checkpoint 8)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from src.assignment import assign_group

    print("=== Pipeline smoke test: scenario1 (clear diff), full flow with sequential peeking ===")
    rng = np.random.default_rng(7)
    n_users = 20000
    p_control, p_treatment = 0.10, 0.15
    experiment_id = "demo-1"

    # 1. Assign every user to a group via consistent hashing (checkpoint 2) --
    #    not simulator.py's internal random split, but the real assignment path.
    user_ids = [f"user_{i}" for i in range(n_users)]
    assigned_groups = [assign_group(uid, experiment_id) for uid in user_ids]

    # 2. Simulate each user's outcome according to the group they were assigned to.
    true_rate = np.where(np.array(assigned_groups) == "A", p_control, p_treatment)
    converted = rng.binomial(1, true_rate)
    user_df = pd.DataFrame({"user_id": user_ids, "group": assigned_groups, "converted": converted})

    config = ExperimentConfig(
        experiment_id=experiment_id,
        primary_metric_column="converted",
        metric_type="proportion",
        baseline_rate=p_control,
        minimum_detectable_effect=p_treatment - p_control,
        max_looks=5,
    )

    # 3. Simulate "peeking" at growing slices of the data -- checkpoint 4's
    #    O'Brien-Fleming boundary should let us do this safely, unlike a static
    #    p<alpha check on every look.
    for look in range(1, config.max_looks + 1):
        n_so_far = int(n_users * look / config.max_looks)
        result = analyze_experiment(user_df.iloc[:n_so_far], config, look_number=look)
        print(f"\n--- Look {look}/{config.max_looks} (n={n_so_far}) ---")
        print(result.summary())
        if result.sequential and result.sequential.recommendation != "continue":
            break

    print("\n=== Guardrail demo: wins primary metric, hurts guardrail (checkpoint 8) ===")
    rng = np.random.default_rng(99)
    n = 20000
    groups = rng.choice(["A", "B"], size=n)
    # primary metric: CTR -- treatment wins
    ctr_true = np.where(groups == "A", 0.10, 0.15)
    converted = rng.binomial(1, ctr_true)
    # guardrail: session duration (minutes) -- treatment secretly hurts it
    duration_mean = np.where(groups == "A", 12.0, 9.0)
    duration = rng.normal(duration_mean, 3.0)

    guardrail_df = pd.DataFrame({"group": groups, "converted": converted, "session_minutes": duration})

    guardrail_config = ExperimentConfig(
        experiment_id="demo-guardrail",
        primary_metric_column="converted",
        metric_type="proportion",
        guardrails=[GuardrailConfig(name="session_minutes", column="session_minutes", metric_type="continuous", lower_is_worse=True)],
    )
    guardrail_result = analyze_experiment(guardrail_df, guardrail_config)
    print(guardrail_result.summary())

    print("\n=== Sequential demo: small effect, shows 'continue' across several looks ===")
    experiment_id_2 = "demo-2"
    n_users_2 = 40000
    p_control_2, p_treatment_2 = 0.10, 0.11
    user_ids_2 = [f"user2_{i}" for i in range(n_users_2)]
    assigned_groups_2 = [assign_group(uid, experiment_id_2) for uid in user_ids_2]
    rng2 = np.random.default_rng(11)
    true_rate_2 = np.where(np.array(assigned_groups_2) == "A", p_control_2, p_treatment_2)
    converted_2 = rng2.binomial(1, true_rate_2)
    user_df_2 = pd.DataFrame({"user_id": user_ids_2, "group": assigned_groups_2, "converted": converted_2})

    config_2 = ExperimentConfig(
        experiment_id=experiment_id_2,
        primary_metric_column="converted",
        metric_type="proportion",
        max_looks=8,
    )
    for look in range(1, config_2.max_looks + 1):
        n_so_far = int(n_users_2 * look / config_2.max_looks)
        result = analyze_experiment(user_df_2.iloc[:n_so_far], config_2, look_number=look)
        print(f"\n--- Look {look}/{config_2.max_looks} (n={n_so_far}) ---")
        print(result.sequential.message)
        if result.sequential.recommendation != "continue":
            break


if __name__ == "__main__":
    main()
