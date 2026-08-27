"""
Power analysis: how many users (per group) does an experiment need before
it can reliably detect a given effect size, and a guard that warns when an
experiment is being read out with too few users.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
    relative: bool = False,
) -> int:
    """Minimum users needed *per group* to detect `minimum_detectable_effect`
    with the given power and significance level.

    `minimum_detectable_effect` is an absolute rate difference by default
    (e.g. 0.003 for 10.0% -> 10.3%); pass relative=True to treat it as a
    fraction of the baseline instead (e.g. 0.03 for a 3% relative lift).
    """
    treatment_rate = (
        baseline_rate * (1 + minimum_detectable_effect) if relative else baseline_rate + minimum_detectable_effect
    )

    effect_size = proportion_effectsize(treatment_rate, baseline_rate)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
    return math.ceil(n)


@dataclass
class SampleSizeCheck:
    n_actual: int
    n_required: int
    is_adequate: bool
    message: str


def check_sample_size(n_actual: int, n_required: int) -> SampleSizeCheck:
    """Warn instead of silently trusting a result computed on too few users."""
    is_adequate = n_actual >= n_required
    if is_adequate:
        message = f"Cỡ mẫu đủ ({n_actual} >= {n_required} yêu cầu)."
    else:
        message = (
            f"CẢNH BÁO: cỡ mẫu hiện tại ({n_actual}) nhỏ hơn cỡ mẫu cần thiết "
            f"({n_required}) để phát hiện hiệu ứng ở power đã cấu hình. Kết luận "
            f"'không có khác biệt' ở giai đoạn này CHƯA đủ tin cậy -- có thể chỉ "
            f"là thiếu mẫu, không phải thực sự không có hiệu ứng."
        )
    return SampleSizeCheck(n_actual=n_actual, n_required=n_required, is_adequate=is_adequate, message=message)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from src.simulator import SCENARIOS, aggregate_to_user_level, simulate_experiment
    from src.stats_engine import run_proportions_test

    rates = SCENARIOS["scenario3_small_diff"]
    p_control, p_treatment = rates["p_control"], rates["p_treatment"]
    mde = p_treatment - p_control
    target_power = 0.8

    n_required = required_sample_size(p_control, mde, power=target_power)
    print(f"Scenario 3 (small diff): p_control={p_control}, p_treatment={p_treatment}, MDE={mde}")
    print(f"Required sample size per group (power={target_power}, alpha=0.05): {n_required}")

    # A single simulated run can go either way by luck (that's what "power" means:
    # even at the required n, ~20% of runs will still miss a real effect). So we
    # validate empirically: repeat many times and check the *detection rate*
    # matches the target power, instead of eyeballing one run's p-value.
    n_replicates = 200
    for label, n_per_group in [
        ("cỡ mẫu QUÁ NHỎ (30% yêu cầu)", int(n_required * 0.3)),
        ("cỡ mẫu ĐỦ (đúng theo tính toán)", n_required),
    ]:
        n_total = n_per_group * 2
        detections = 0
        for i in range(n_replicates):
            df = simulate_experiment(n_total, p_control, p_treatment, seed=1000 + i)
            user_df = aggregate_to_user_level(df)
            control = user_df[user_df.group == "A"]
            treatment = user_df[user_df.group == "B"]

            result = run_proportions_test(
                conversions_control=int(control.converted.sum()),
                n_control=len(control),
                conversions_treatment=int(treatment.converted.sum()),
                n_treatment=len(treatment),
            )
            detections += result.is_significant

        empirical_power = detections / n_replicates
        check = check_sample_size(n_total, n_required * 2)

        print(f"\n[{label}] n/group={n_per_group}")
        print(f"  Empirical power over {n_replicates} runs: {empirical_power:.2f} (detected {detections}/{n_replicates})")
        print(f"  {check.message}")


if __name__ == "__main__":
    main()
