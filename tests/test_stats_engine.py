import numpy as np

from src.simulator import aggregate_to_user_level, simulate_experiment
from src.stats_engine import run_proportions_test, run_ttest


def test_detects_clear_true_difference():
    df = simulate_experiment(20000, p_control=0.10, p_treatment=0.15, seed=1)
    user_df = aggregate_to_user_level(df)
    control = user_df[user_df.group == "A"]
    treatment = user_df[user_df.group == "B"]

    result = run_proportions_test(
        conversions_control=int(control.converted.sum()),
        n_control=len(control),
        conversions_treatment=int(treatment.converted.sum()),
        n_treatment=len(treatment),
    )
    assert result.is_significant
    assert result.diff > 0


def test_does_not_falsely_detect_no_difference():
    df = simulate_experiment(20000, p_control=0.10, p_treatment=0.10, seed=2)
    user_df = aggregate_to_user_level(df)
    control = user_df[user_df.group == "A"]
    treatment = user_df[user_df.group == "B"]

    result = run_proportions_test(
        conversions_control=int(control.converted.sum()),
        n_control=len(control),
        conversions_treatment=int(treatment.converted.sum()),
        n_treatment=len(treatment),
    )
    assert not result.is_significant


def test_ttest_detects_continuous_difference():
    rng = np.random.default_rng(3)
    control = rng.normal(10.0, 2.0, size=2000)
    treatment = rng.normal(11.0, 2.0, size=2000)

    result = run_ttest(control, treatment)
    assert result.is_significant
    assert result.diff > 0
