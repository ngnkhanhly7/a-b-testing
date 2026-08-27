"""
Deterministic user -> group assignment via consistent hashing, plus a
Sample Ratio Mismatch (SRM) check to catch a broken/biased assignment path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from scipy.stats import chisquare


def assign_group(user_id: str | int, experiment_id: str, split: dict[str, float] | None = None) -> str:
    """Deterministically assign a user to a group.

    Same (user_id, experiment_id) always yields the same group, without
    storing any state, by hashing the pair into [0, 1) and bucketing
    according to `split` (defaults to 50/50 A/B).
    """
    split = split or {"A": 0.5, "B": 0.5}

    key = f"{experiment_id}:{user_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # -> [0, 1)

    cumulative = 0.0
    for group, weight in split.items():
        cumulative += weight
        if bucket < cumulative:
            return group
    return list(split.keys())[-1]  # floating point safety net


@dataclass
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    is_mismatched: bool


def check_srm(group_counts: dict[str, int], expected_split: dict[str, float] | None = None, alpha: float = 0.001) -> SRMResult:
    """Sample Ratio Mismatch check.

    A low p-value (below `alpha`) means the observed group split deviates
    from the configured split by more than chance would explain -- a strong
    signal of a broken assignment/logging pipeline. `alpha` defaults to a
    strict 0.001 (not 0.05) because SRM checks run on every experiment and a
    5% false-alarm rate would be too noisy in practice; this is the standard
    convention used by SRM checkers at large A/B platforms.
    """
    if expected_split is None:
        expected_split = {g: 1.0 / len(group_counts) for g in group_counts}

    # Use expected_split as the source of truth for which groups exist, not
    # just the groups seen in group_counts -- otherwise a group with zero
    # observed users so far (common early in a live experiment) silently
    # drops out of the comparison instead of showing up as a glaring mismatch.
    groups = list(expected_split.keys())
    n_total = sum(group_counts.values())

    observed = [group_counts.get(g, 0) for g in groups]
    expected = [expected_split[g] * n_total for g in groups]

    chi2, p_value = chisquare(f_obs=observed, f_exp=expected)

    return SRMResult(
        observed={g: int(v) for g, v in zip(groups, observed)},
        expected={g: float(v) for g, v in zip(groups, expected)},
        chi2=float(chi2),
        p_value=float(p_value),
        is_mismatched=bool(p_value < alpha),
    )
