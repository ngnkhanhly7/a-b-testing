"""
Short, presentation-friendly demo for the A/B testing platform.

Run:
    python -m src.demo_showcase

The demo uses simulated data with known ground truth, then sends it through
the same pipeline used by the API. It is meant for showing the project to
someone who does not want to click through Swagger or read raw test output.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.assignment import assign_group
from src.pipeline import ExperimentConfig, GuardrailConfig, analyze_experiment


def _headline(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def _simulate_users(
    experiment_id: str,
    n_users: int,
    p_control: float,
    p_treatment: float,
    seed: int,
    include_guardrail: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = [f"user_{i}" for i in range(n_users)]
    groups = np.array([assign_group(uid, experiment_id) for uid in user_ids])
    converted = rng.binomial(1, np.where(groups == "A", p_control, p_treatment))

    df = pd.DataFrame({"user_id": user_ids, "group": groups, "converted": converted})
    if include_guardrail:
        # Treatment converts better, but makes sessions shorter. This shows why
        # winning the primary metric is not always enough to ship safely.
        df["session_minutes"] = rng.normal(np.where(groups == "A", 12.0, 9.0), 3.0)
    return df


def _print_rates(df: pd.DataFrame) -> None:
    rates = df.groupby("group")["converted"].agg(users="count", conversion_rate="mean")
    print(rates.to_string(formatters={"conversion_rate": lambda x: f"{x:.2%}"}))


def demo_clear_winner() -> None:
    _headline("DEMO 1: Treatment wins clearly")
    print("Story: Control converts around 10%, treatment converts around 15%.")
    print("Question: Does the platform detect the lift and tell us it is safe?")

    df = _simulate_users("showcase-clear", 20_000, 0.10, 0.15, seed=7)
    _print_rates(df)

    config = ExperimentConfig(
        experiment_id="showcase-clear",
        primary_metric_column="converted",
        metric_type="proportion",
        baseline_rate=0.10,
        minimum_detectable_effect=0.05,
        max_looks=5,
    )

    for look in range(1, config.max_looks + 1):
        n_so_far = int(len(df) * look / config.max_looks)
        result = analyze_experiment(df.iloc[:n_so_far], config, look_number=look)
        print(f"\nLook {look}/{config.max_looks}, users so far: {n_so_far}")
        print(result.primary.conclusion)
        print(result.sequential.message)
        if result.sequential.recommendation != "continue":
            print("\nTakeaway: the test can stop early because the effect is strong.")
            break


def demo_no_difference() -> None:
    _headline("DEMO 2: No real difference")
    print("Story: Both variants truly convert around 10%.")
    print("Question: Does the platform avoid claiming a fake winner?")

    df = _simulate_users("showcase-no-diff", 20_000, 0.10, 0.10, seed=42)
    _print_rates(df)

    config = ExperimentConfig(
        experiment_id="showcase-no-diff",
        primary_metric_column="converted",
        metric_type="proportion",
        max_looks=5,
    )
    result = analyze_experiment(df, config, look_number=5)

    print("\nFinal result:")
    print(result.primary.conclusion)
    print(result.sequential.message)
    print("\nTakeaway: no fake victory; the platform says there is not enough evidence.")


def demo_guardrail() -> None:
    _headline("DEMO 3: Primary metric wins, guardrail gets worse")
    print("Story: Treatment improves conversion, but reduces session duration.")
    print("Question: Does the platform warn us before we ship a harmful variant?")

    df = _simulate_users("showcase-guardrail", 20_000, 0.10, 0.15, seed=99, include_guardrail=True)
    _print_rates(df)
    print(
        "\nAverage session minutes:\n"
        + df.groupby("group")["session_minutes"].mean().to_string(float_format=lambda x: f"{x:.2f}")
    )

    config = ExperimentConfig(
        experiment_id="showcase-guardrail",
        primary_metric_column="converted",
        metric_type="proportion",
        guardrails=[
            GuardrailConfig(
                name="session_minutes",
                column="session_minutes",
                metric_type="continuous",
                lower_is_worse=True,
            )
        ],
    )
    result = analyze_experiment(df, config)

    print("\nPlatform summary:")
    print(result.summary())
    print("\nTakeaway: a higher conversion rate is not enough if a guardrail regresses.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("A/B Testing Platform - simple showcase demo")
    print("This uses fake data where we know the truth, then checks if the platform reaches the right conclusion.")
    demo_clear_winner()
    demo_no_difference()
    demo_guardrail()


if __name__ == "__main__":
    main()
