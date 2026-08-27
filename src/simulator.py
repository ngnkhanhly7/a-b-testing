"""
Traffic simulator with known ground truth.

Every downstream module (assignment, stats_engine, sequential, power) is
validated against the scenarios produced here, because we know the *true*
conversion rates going in and can therefore check whether the tools recover
the right answer.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DATE = datetime(2026, 1, 1)

SCENARIOS = {
    "scenario1_clear_diff": dict(p_control=0.10, p_treatment=0.15),
    "scenario2_no_diff": dict(p_control=0.10, p_treatment=0.10),
    "scenario3_small_diff": dict(p_control=0.10, p_treatment=0.103),
}


def simulate_experiment(
    n_users: int,
    p_control: float,
    p_treatment: float,
    days: int = 14,
    seed: int | None = 42,
    uneven_traffic: bool = False,
    multi_session: bool = False,
) -> pd.DataFrame:
    """Generate an event log with a known ground-truth conversion rate per group.

    Returns a DataFrame with columns: user_id, group, event, timestamp.
    `event` is 1 for a conversion, 0 for a plain visit/session with no conversion.
    A user can appear in multiple rows when `multi_session=True` (realistic
    noise: not every user maps to exactly one row), so downstream consumers
    must aggregate to user level (see `aggregate_to_user_level`) before running
    a per-user statistical test.
    """
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)
    groups = rng.choice(["A", "B"], size=n_users, p=[0.5, 0.5])

    if uneven_traffic:
        day_weights = rng.dirichlet(np.full(days, 2.0))
    else:
        day_weights = np.full(days, 1.0 / days)
    signup_days = rng.choice(np.arange(days), size=n_users, p=day_weights)

    true_rate = np.where(groups == "A", p_control, p_treatment)
    converted = rng.binomial(1, true_rate)

    base_ts = pd.Timestamp(BASE_DATE) + pd.to_timedelta(signup_days, unit="D") + pd.to_timedelta(
        rng.integers(0, 86400, size=n_users), unit="s"
    )
    base_df = pd.DataFrame({"user_id": user_ids, "group": groups, "event": converted, "timestamp": base_ts})

    if not multi_session:
        return base_df.sort_values("timestamp").reset_index(drop=True)

    # Vectorized extra (non-converting) sessions: draw how many extra visits
    # each user gets, then expand into one row per extra visit in one shot
    # instead of a per-user Python loop -- this dominated runtime at scale
    # (e.g. the Monte Carlo power validation replays this hundreds of times).
    n_extra = rng.poisson(0.5, size=n_users)
    total_extra = int(n_extra.sum())
    if total_extra == 0:
        return base_df.sort_values("timestamp").reset_index(drop=True)

    extra_idx = np.repeat(np.arange(n_users), n_extra)
    extra_signup_days = signup_days[extra_idx]
    max_offset = np.maximum(days - extra_signup_days, 1)
    extra_day = np.minimum(extra_signup_days + rng.integers(0, max_offset), days - 1)
    extra_ts = pd.Timestamp(BASE_DATE) + pd.to_timedelta(extra_day, unit="D") + pd.to_timedelta(
        rng.integers(0, 86400, size=total_extra), unit="s"
    )
    extra_df = pd.DataFrame(
        {
            "user_id": user_ids[extra_idx],
            "group": groups[extra_idx],
            "event": 0,
            "timestamp": extra_ts,
        }
    )

    return pd.concat([base_df, extra_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def aggregate_to_user_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (possibly multi-session) event log to one row per user.

    A user counts as converted if *any* of their sessions converted.
    """
    return (
        df.groupby(["user_id", "group"], as_index=False)["event"]
        .max()
        .rename(columns={"event": "converted"})
    )


def _print_actual_rates(name: str, df: pd.DataFrame) -> None:
    user_df = aggregate_to_user_level(df)
    rates = user_df.groupby("group")["converted"].agg(["mean", "count"])
    print(f"\n[{name}]")
    print(rates.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated A/B test scenarios with known ground truth.")
    parser.add_argument("--n-users", type=int, default=20000)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out-dir", default="data/simulated")
    parser.add_argument("--uneven-traffic", action="store_true")
    parser.add_argument("--multi-session", action="store_true")
    args = parser.parse_args()

    import os

    os.makedirs(args.out_dir, exist_ok=True)

    for name, rates in SCENARIOS.items():
        df = simulate_experiment(
            n_users=args.n_users,
            p_control=rates["p_control"],
            p_treatment=rates["p_treatment"],
            days=args.days,
            uneven_traffic=args.uneven_traffic,
            multi_session=args.multi_session,
        )
        out_path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)
        _print_actual_rates(name, df)
        print(f"  -> saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
