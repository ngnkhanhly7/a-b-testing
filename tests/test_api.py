import pytest
from fastapi import HTTPException

from src import storage
from src.api.main import (
    EventIn,
    ExperimentIn,
    ExtendMaxLooksIn,
    create_experiment,
    extend_max_looks,
    get_results,
    log_event,
    verify_experiment_access,
)


def test_results_reports_insufficient_data_when_one_group_empty():
    exp = create_experiment(ExperimentIn(name="edge", metric_type="proportion"))
    exp_id = exp["experiment_id"]
    log_event(exp_id, EventIn(user_id="u1", group="A", metrics={"converted": 1}))

    result = get_results(exp_id)
    assert result["status"] == "insufficient_data"


def test_sequential_recommendation_progresses_to_stop():
    exp = create_experiment(ExperimentIn(name="seq", metric_type="proportion", max_looks=5))
    exp_id = exp["experiment_id"]

    import numpy as np

    rng = np.random.default_rng(11)
    n_total = 25000
    p_control, p_treatment = 0.10, 0.11

    recommendations = []
    for batch in range(5):
        start = batch * (n_total // 5)
        end = (batch + 1) * (n_total // 5)
        for i in range(start, end):
            group = "A" if i % 2 == 0 else "B"
            p = p_control if group == "A" else p_treatment
            conv = int(rng.random() < p)
            log_event(exp_id, EventIn(user_id=f"u{i}", group=group, metrics={"converted": conv}))
        result = get_results(exp_id)
        recommendations.append(result["sequential"]["recommendation"])
        if result["sequential"]["recommendation"] != "continue":
            break  # experiment is now locked -- no more events would be accepted

    assert recommendations[0] == "continue"
    assert recommendations[-1] in {"stop_significant", "stop_no_effect"}


def test_locked_experiment_rejects_further_events_and_freezes_result():
    exp = create_experiment(ExperimentIn(name="lock-test", metric_type="proportion", max_looks=2))
    exp_id = exp["experiment_id"]

    for i in range(4000):
        # deterministic (not random) so the test is fast and reproducible;
        # both groups convert sometimes so relative_diff_pct isn't NaN, which
        # would otherwise break the equality check below (NaN != NaN).
        group = "A" if i % 2 == 0 else "B"
        converted = int(i % 20 == 0) if group == "A" else int(i % 5 == 0)
        log_event(exp_id, EventIn(user_id=f"u{i}", group=group, metrics={"converted": converted}))

    result = get_results(exp_id)
    assert result["locked"] is True

    with pytest.raises(HTTPException) as exc_info:
        log_event(exp_id, EventIn(user_id="late_user", group="A", metrics={"converted": 0}))
    assert exc_info.value.status_code == 409

    # reading results again must return the same frozen verdict, not a fresh one
    result_again = get_results(exp_id)
    assert result_again == result


def test_extend_max_looks_raises_the_budget():
    exp = create_experiment(ExperimentIn(name="extend-test", metric_type="proportion", max_looks=2))
    exp_id = exp["experiment_id"]

    response = extend_max_looks(exp_id, ExtendMaxLooksIn(max_looks=10))
    assert response["max_looks"] == 10


def test_extend_max_looks_rejects_non_increasing_value():
    exp = create_experiment(ExperimentIn(name="extend-reject", metric_type="proportion", max_looks=5))
    exp_id = exp["experiment_id"]

    with pytest.raises(HTTPException) as exc_info:
        extend_max_looks(exp_id, ExtendMaxLooksIn(max_looks=5))
    assert exc_info.value.status_code == 400


def test_create_experiment_returns_a_per_experiment_key():
    exp = create_experiment(ExperimentIn(name="scoped", metric_type="proportion"))
    assert exp["experiment_key"]
    assert exp["experiment_key"] != storage.get_or_create_api_key()


def test_verify_experiment_access_accepts_admin_key_and_own_key_only():
    exp = create_experiment(ExperimentIn(name="scoped-access", metric_type="proportion"))
    exp_id = exp["experiment_id"]
    admin_key = storage.get_or_create_api_key()

    verify_experiment_access(exp_id, admin_key)  # does not raise
    verify_experiment_access(exp_id, exp["experiment_key"])  # does not raise

    with pytest.raises(HTTPException) as exc_info:
        verify_experiment_access(exp_id, "wrong-key")
    assert exc_info.value.status_code == 401

    with pytest.raises(HTTPException) as exc_info:
        verify_experiment_access(exp_id, None)
    assert exc_info.value.status_code == 401


def test_verify_experiment_access_rejects_another_experiments_key():
    exp1 = create_experiment(ExperimentIn(name="scoped-1", metric_type="proportion"))
    exp2 = create_experiment(ExperimentIn(name="scoped-2", metric_type="proportion"))

    with pytest.raises(HTTPException) as exc_info:
        verify_experiment_access(exp1["experiment_id"], exp2["experiment_key"])
    assert exc_info.value.status_code == 401


def test_extend_max_looks_rejects_locked_experiment():
    exp = create_experiment(ExperimentIn(name="extend-locked", metric_type="proportion", max_looks=2))
    exp_id = exp["experiment_id"]

    for i in range(4000):
        group = "A" if i % 2 == 0 else "B"
        converted = int(i % 20 == 0) if group == "A" else int(i % 5 == 0)
        log_event(exp_id, EventIn(user_id=f"u{i}", group=group, metrics={"converted": converted}))
    get_results(exp_id)  # drives look_count up to the locking look

    with pytest.raises(HTTPException) as exc_info:
        extend_max_looks(exp_id, ExtendMaxLooksIn(max_looks=20))
    assert exc_info.value.status_code == 409
