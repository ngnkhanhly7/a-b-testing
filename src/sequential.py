"""
Peeking demonstration + a group-sequential correction (O'Brien-Fleming-style
boundaries) that keeps repeated looks at the data under control.

The core problem: if you check p-value every day and stop the moment
p < 0.05, your true false-positive rate is much higher than 5%, even when
there is truly no difference between groups. This module first proves that
with simulation, then shows a boundary scheme that fixes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def sequential_z_stat(conversions_control: int, n_control: int, conversions_treatment: int, n_treatment: int) -> float:
    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    if se == 0:
        return 0.0
    return (p_treatment - p_control) / se


def obrien_fleming_boundary(look: int, n_looks: int, alpha: float = 0.05) -> float:
    """Classic O'Brien-Fleming approximate boundary: z_crit(k) = z_{alpha/2} * sqrt(K/k).

    Very conservative early on, converging to the standard z_{alpha/2}
    critical value at the final look -- this is what lets you peek often
    without inflating the overall false-positive rate.
    """
    z_alpha2 = norm.ppf(1 - alpha / 2)
    return z_alpha2 * np.sqrt(n_looks / look)


@dataclass
class PeekingSimResult:
    method: str
    n_looks: int
    n_simulations: int
    nominal_alpha: float
    false_positive_rate: float


def simulate_false_positive_rate(
    p_control: float,
    p_treatment: float,
    n_per_look: int,
    n_looks: int,
    n_simulations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    use_obf: bool = False,
) -> PeekingSimResult:
    """Run many fake experiments and measure how often the stopping rule
    incorrectly declares significance.

    Pass p_control == p_treatment to measure the *false*-positive rate
    (there is no true effect); the naive "stop as soon as p<0.05" rule
    should show a rate well above `alpha`, while use_obf=True should bring
    it back down close to `alpha`.
    """
    rng = np.random.default_rng(seed)
    z_alpha2 = norm.ppf(1 - alpha / 2)

    false_positives = 0
    for _ in range(n_simulations):
        conv_c = n_c = conv_t = n_t = 0
        stopped_significant = False
        for look in range(1, n_looks + 1):
            conv_c += int(rng.binomial(n_per_look, p_control))
            conv_t += int(rng.binomial(n_per_look, p_treatment))
            n_c += n_per_look
            n_t += n_per_look

            z = sequential_z_stat(conv_c, n_c, conv_t, n_t)
            boundary = obrien_fleming_boundary(look, n_looks, alpha) if use_obf else z_alpha2

            if abs(z) > boundary:
                stopped_significant = True
                break
        if stopped_significant:
            false_positives += 1

    return PeekingSimResult(
        method="O'Brien-Fleming (alpha-spending)" if use_obf else "Naive peeking (fixed alpha every look)",
        n_looks=n_looks,
        n_simulations=n_simulations,
        nominal_alpha=alpha,
        false_positive_rate=false_positives / n_simulations,
    )


def main() -> None:
    p_null = 0.10  # scenario2: no true difference
    n_per_look = 200
    n_looks = 10
    n_simulations = 3000
    alpha = 0.05

    naive = simulate_false_positive_rate(
        p_null, p_null, n_per_look, n_looks, n_simulations, alpha, use_obf=False
    )
    corrected = simulate_false_positive_rate(
        p_null, p_null, n_per_look, n_looks, n_simulations, alpha, use_obf=True
    )

    print(f"Nominal alpha: {alpha}")
    print(f"{naive.method}: false positive rate = {naive.false_positive_rate:.4f}")
    print(f"{corrected.method}: false positive rate = {corrected.false_positive_rate:.4f}")

    report = (
        "# Sequential Testing Validation\n\n"
        f"Scenario: no true difference (p_control = p_treatment = {p_null}), "
        f"{n_looks} looks of {n_per_look} users/group each, {n_simulations} simulated experiments.\n\n"
        "| Method | False positive rate | Nominal alpha |\n"
        "|---|---|---|\n"
        f"| {naive.method} | {naive.false_positive_rate:.4f} | {alpha} |\n"
        f"| {corrected.method} | {corrected.false_positive_rate:.4f} | {alpha} |\n\n"
        "Naive peeking inflates the false-positive rate well above the nominal "
        "alpha because every daily check is an independent chance to cross the "
        "p<0.05 threshold by chance alone. The O'Brien-Fleming boundary spends "
        "the alpha budget across looks (very strict early, relaxing to the "
        "standard critical value only at the final look), bringing the false "
        "positive rate back down close to the nominal alpha.\n"
    )
    import os

    os.makedirs("reports", exist_ok=True)
    with open("reports/sequential_validation.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nSaved reports/sequential_validation.md")


if __name__ == "__main__":
    main()
