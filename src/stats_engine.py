"""
Core statistical test engine: two-proportion z-test (rates) and Welch's
t-test (continuous metrics), both returning a p-value, a 95% CI for the
difference, an effect size, and a plain-language conclusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class TestResult:
    metric_type: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    diff: float  # treatment - control
    relative_diff_pct: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    is_significant: bool
    conclusion: str


def _conclusion_text(result_kwargs: dict) -> str:
    diff = result_kwargs["diff"]
    rel = result_kwargs["relative_diff_pct"]
    ci_low, ci_high = result_kwargs["ci_low"], result_kwargs["ci_high"]
    alpha = result_kwargs["alpha"]

    if not result_kwargs["is_significant"]:
        return (
            f"Không đủ bằng chứng để kết luận có khác biệt giữa 2 nhóm "
            f"(p={result_kwargs['p_value']:.4f} >= alpha={alpha})."
        )

    direction = "cao hơn" if diff > 0 else "thấp hơn"
    return (
        f"Treatment {direction} Control {abs(rel):.2f}% (chênh lệch tuyệt đối "
        f"{diff:+.4f}), tin cậy {(1 - alpha) * 100:.0f}%, "
        f"CI: [{ci_low:+.4f}, {ci_high:+.4f}]."
    )


def run_proportions_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-proportion z-test, for rate metrics (conversion, click, retention)."""
    count = np.array([conversions_treatment, conversions_control])
    nobs = np.array([n_treatment, n_control])

    z_stat, p_value = proportions_ztest(count, nobs)

    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment
    diff = p_treatment - p_control

    se = np.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - z_crit * se, diff + z_crit * se

    relative_diff_pct = (diff / p_control * 100) if p_control > 0 else float("nan")

    kwargs = dict(
        metric_type="proportion",
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        mean_control=float(p_control),
        mean_treatment=float(p_treatment),
        diff=float(diff),
        relative_diff_pct=float(relative_diff_pct),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=bool(p_value < alpha),
    )
    kwargs["conclusion"] = _conclusion_text(kwargs)
    return TestResult(**kwargs)


def run_ttest(
    values_control: np.ndarray,
    values_treatment: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test, for continuous metrics (revenue, session duration)."""
    values_control = np.asarray(values_control, dtype=float)
    values_treatment = np.asarray(values_treatment, dtype=float)

    t_stat, p_value = scipy_stats.ttest_ind(values_treatment, values_control, equal_var=False)

    mean_control = values_control.mean()
    mean_treatment = values_treatment.mean()
    diff = mean_treatment - mean_control

    var_over_n_control = values_control.var(ddof=1) / len(values_control)
    var_over_n_treatment = values_treatment.var(ddof=1) / len(values_treatment)

    se = np.sqrt(var_over_n_control + var_over_n_treatment)
    # Welch-Satterthwaite degrees of freedom
    dof = (var_over_n_control + var_over_n_treatment) ** 2 / (
        var_over_n_control**2 / (len(values_control) - 1) + var_over_n_treatment**2 / (len(values_treatment) - 1)
    )
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, dof)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    relative_diff_pct = (diff / mean_control * 100) if mean_control != 0 else float("nan")

    kwargs = dict(
        metric_type="continuous",
        n_control=len(values_control),
        n_treatment=len(values_treatment),
        mean_control=float(mean_control),
        mean_treatment=float(mean_treatment),
        diff=float(diff),
        relative_diff_pct=float(relative_diff_pct),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        alpha=alpha,
        is_significant=bool(p_value < alpha),
    )
    kwargs["conclusion"] = _conclusion_text(kwargs)
    return TestResult(**kwargs)


def run_test(data, metric_type: str, alpha: float = 0.05) -> TestResult:
    """Dispatch entry point.

    `data` for metric_type="proportion" is a dict with keys
    conversions_control, n_control, conversions_treatment, n_treatment.
    `data` for metric_type="continuous" is a dict with keys
    values_control, values_treatment (array-likes).
    """
    if metric_type == "proportion":
        return run_proportions_test(alpha=alpha, **data)
    if metric_type == "continuous":
        return run_ttest(alpha=alpha, **data)
    raise ValueError(f"Unknown metric_type: {metric_type}")
