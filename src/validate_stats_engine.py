"""
Checkpoint 3 validation: run stats_engine on the 3 known-ground-truth
scenarios from src/simulator.py and check the tool reaches the right
conclusion on each, writing the comparison to reports/stats_engine_validation.md.

Run `python -m src.simulator` first to generate the scenario CSVs.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src.simulator import SCENARIOS, aggregate_to_user_level
from src.stats_engine import run_proportions_test

DATA_DIR = "data/simulated"
REPORT_PATH = "reports/stats_engine_validation.md"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    for name, rates in SCENARIOS.items():
        csv_path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.simulator` first.")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        user_df = aggregate_to_user_level(df)
        control = user_df[user_df.group == "A"]
        treatment = user_df[user_df.group == "B"]

        result = run_proportions_test(
            conversions_control=int(control.converted.sum()),
            n_control=len(control),
            conversions_treatment=int(treatment.converted.sum()),
            n_treatment=len(treatment),
        )

        rows.append(
            {
                "scenario": name,
                "true_p_control": rates["p_control"],
                "true_p_treatment": rates["p_treatment"],
                "observed_p_control": result.mean_control,
                "observed_p_treatment": result.mean_treatment,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "conclusion": result.conclusion,
            }
        )

    df_report = pd.DataFrame(rows)

    lines = ["# Stats Engine Validation\n"]
    for _, r in df_report.iterrows():
        lines.append(f"## {r['scenario']}")
        lines.append(f"- True rates: control={r['true_p_control']}, treatment={r['true_p_treatment']}")
        lines.append(f"- Observed rates: control={r['observed_p_control']:.4f}, treatment={r['observed_p_treatment']:.4f}")
        lines.append(f"- p-value: {r['p_value']:.4g}, significant: {r['is_significant']}")
        lines.append(f"- Conclusion: {r['conclusion']}\n")

    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
