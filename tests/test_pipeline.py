import numpy as np
import pandas as pd

from src.assignment import assign_group
from src.pipeline import ExperimentConfig, GuardrailConfig, analyze_experiment


def _make_user_df(experiment_id, n_users, p_control, p_treatment, seed):
    user_ids = [f"user_{i}" for i in range(n_users)]
    groups = [assign_group(uid, experiment_id) for uid in user_ids]
    rng = np.random.default_rng(seed)
    true_rate = np.where(np.array(groups) == "A", p_control, p_treatment)
    converted = rng.binomial(1, true_rate)
    return pd.DataFrame({"user_id": user_ids, "group": groups, "converted": converted})


def test_sequential_recommendation_continues_then_stops():
    experiment_id = "seq-test"
    user_df = _make_user_df(experiment_id, 40000, p_control=0.10, p_treatment=0.11, seed=11)
    config = ExperimentConfig(
        experiment_id=experiment_id,
        primary_metric_column="converted",
        metric_type="proportion",
        max_looks=8,
    )

    recommendations = []
    for look in range(1, config.max_looks + 1):
        n_so_far = int(len(user_df) * look / config.max_looks)
        result = analyze_experiment(user_df.iloc[:n_so_far], config, look_number=look)
        recommendations.append(result.sequential.recommendation)
        if result.sequential.recommendation != "continue":
            break

    assert recommendations[0] == "continue"
    assert recommendations[-1] in {"stop_significant", "stop_no_effect"}


def test_no_sequential_check_without_max_looks():
    experiment_id = "no-seq-test"
    user_df = _make_user_df(experiment_id, 2000, p_control=0.10, p_treatment=0.15, seed=1)
    config = ExperimentConfig(
        experiment_id=experiment_id,
        primary_metric_column="converted",
        metric_type="proportion",
    )
    result = analyze_experiment(user_df, config)
    assert result.sequential is None


def test_guardrail_regression_detected_alongside_primary_win():
    rng = np.random.default_rng(99)
    n = 20000
    groups = rng.choice(["A", "B"], size=n)
    converted = rng.binomial(1, np.where(groups == "A", 0.10, 0.15))
    duration = rng.normal(np.where(groups == "A", 12.0, 9.0), 3.0)
    df = pd.DataFrame({"group": groups, "converted": converted, "session_minutes": duration})

    config = ExperimentConfig(
        experiment_id="guardrail-test",
        primary_metric_column="converted",
        metric_type="proportion",
        guardrails=[GuardrailConfig(name="session_minutes", column="session_minutes", metric_type="continuous")],
    )
    result = analyze_experiment(df, config)

    assert result.primary.is_significant and result.primary.diff > 0
    assert result.guardrails["session_minutes"]["regression_detected"]
