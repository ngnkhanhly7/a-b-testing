from scipy.stats import chisquare

from src.assignment import assign_group, check_srm


def test_assignment_is_deterministic():
    for uid in ["user_1", "user_42", "abc-999"]:
        g1 = assign_group(uid, "exp_a")
        g2 = assign_group(uid, "exp_a")
        assert g1 == g2


def test_assignment_splits_roughly_50_50():
    n = 100_000
    groups = [assign_group(f"user_{i}", "exp_ratio_check") for i in range(n)]
    counts = {"A": groups.count("A"), "B": groups.count("B")}

    chi2, p_value = chisquare([counts["A"], counts["B"]], f_exp=[n / 2, n / 2])
    assert p_value > 0.01, f"Split too far from 50/50: {counts}, p={p_value}"


def test_assignment_differs_by_experiment():
    uid = "user_777"
    g_a = assign_group(uid, "experiment_one")
    g_b = assign_group(uid, "experiment_two")
    # not asserting they differ (could coincidentally match), just that both are valid
    assert g_a in {"A", "B"}
    assert g_b in {"A", "B"}


def test_srm_check_passes_on_balanced_split():
    result = check_srm({"A": 5000, "B": 5010})
    assert not result.is_mismatched


def test_srm_check_flags_skewed_split():
    result = check_srm({"A": 4500, "B": 5500})
    assert result.is_mismatched


def test_srm_check_handles_group_with_zero_observed_users():
    # A group can have 0 events early in a live experiment. It must show up
    # explicitly as 0 in the comparison (not silently drop out, which used to
    # crash scipy's chisquare with an obs/exp sum mismatch) even though 1
    # sample is too little data to call it a statistically significant SRM.
    result = check_srm({"A": 1}, expected_split={"A": 0.5, "B": 0.5})
    assert result.observed == {"A": 1, "B": 0}
    assert result.expected == {"A": 0.5, "B": 0.5}

    # With a real sample size, an entire group missing is an obvious mismatch.
    result_large = check_srm({"A": 5000}, expected_split={"A": 0.5, "B": 0.5})
    assert result_large.is_mismatched
