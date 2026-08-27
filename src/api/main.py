"""
FastAPI layer: create an experiment, log events, read back live statistical
results.

Persisted to SQLite (src/storage.py) -- data survives restarts. Two levels of
auth: a single global admin API key (generated once, stored in the DB) is
required to create experiments, and each experiment also gets its own key
returned once at creation time -- a team can hand that key to teammates or
services without also handing out admin access to every other team's
experiments. Either key works on that experiment's events/results/extend
endpoints. Once sequential testing recommends stopping, the experiment is
locked: no more events are accepted and /results always returns the same
final verdict, so a team can't keep peeking past the point the boundary said
was safe to stop at.

Run with: uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import json
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src import storage
from src.assignment import assign_group
from src.pipeline import ExperimentConfig, GuardrailConfig, analyze_experiment


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    api_key = storage.get_or_create_api_key()
    print(f"[A/B Testing Platform] API key (send as 'X-API-Key' header on every request): {api_key}", flush=True)
    yield


app = FastAPI(title="A/B Testing Platform", lifespan=lifespan)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
    """Admin-only endpoints (currently just POST /experiments): only the
    global key works here, not a per-experiment key."""
    expected = storage.get_or_create_api_key()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def verify_experiment_access(experiment_id: str, x_api_key: str | None = Depends(api_key_header)) -> None:
    """Per-experiment endpoints (events/results/extend): accepts either the
    global admin key or this specific experiment's own key, so a team that
    only has their experiment's key can't read or write any other team's
    experiment."""
    row = storage.get_experiment(experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    admin_key = storage.get_or_create_api_key()
    own_key = row["experiment_key"]
    authorized = x_api_key is not None and (
        secrets.compare_digest(x_api_key, admin_key)
        or (own_key is not None and secrets.compare_digest(x_api_key, own_key))
    )
    if not authorized:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


class GuardrailIn(BaseModel):
    name: str
    column: str
    metric_type: Literal["proportion", "continuous"]
    lower_is_worse: bool = True


class ExperimentIn(BaseModel):
    name: str
    metric_type: Literal["proportion", "continuous"]
    primary_metric_column: str = "converted"
    split: dict[str, float] = Field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    alpha: float = 0.05
    baseline_rate: float | None = None
    minimum_detectable_effect: float | None = None
    guardrails: list[GuardrailIn] = Field(default_factory=list)
    # Planned number of times results will be checked over the experiment's
    # lifetime (checkpoint 4). Every GET /results call counts as one look and
    # is checked against an O'Brien-Fleming boundary instead of a static
    # p<alpha threshold, so checking in on a live experiment repeatedly does
    # not inflate the false-positive rate. Only applies when metric_type is
    # "proportion". Set to null to disable and fall back to a static test.
    max_looks: int | None = 20


class EventIn(BaseModel):
    user_id: str
    group: Literal["A", "B"] | None = None  # if omitted, server assigns deterministically
    metrics: dict[str, float] = Field(default_factory=dict)  # e.g. {"converted": 1, "session_minutes": 4.2}


class ExtendMaxLooksIn(BaseModel):
    max_looks: int


DEMO_HTML = (Path(__file__).parent / "templates" / "demo.html").read_text(encoding="utf-8")


def _config_from_row(row) -> ExperimentConfig:
    data = json.loads(row["config_json"])
    data["guardrails"] = [GuardrailConfig(**g) for g in data["guardrails"]]
    return ExperimentConfig(**data)


def _demo_users(
    experiment_id: str,
    n_users: int,
    p_control: float,
    p_treatment: float,
    seed: int,
    include_guardrail: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = [f"demo_user_{i}" for i in range(n_users)]
    groups = np.array([assign_group(uid, experiment_id) for uid in user_ids])
    converted = rng.binomial(1, np.where(groups == "A", p_control, p_treatment))
    df = pd.DataFrame({"user_id": user_ids, "group": groups, "converted": converted})
    if include_guardrail:
        df["session_minutes"] = rng.normal(np.where(groups == "A", 12.0, 9.0), 3.0)
    return df


def _rates_for(df: pd.DataFrame) -> dict:
    by_group = df.groupby("group")["converted"].mean().to_dict()
    return {"control": float(by_group.get("A", 0.0)), "treatment": float(by_group.get("B", 0.0))}


def _scenario_payload(
    name: str,
    story: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
    badge: str,
    badge_text: str,
    side_title: str,
) -> dict:
    result = analyze_experiment(df, config, look_number=config.max_looks if config.max_looks else None)
    rates = _rates_for(df)
    lift_pct = ((rates["treatment"] - rates["control"]) / rates["control"] * 100) if rates["control"] else 0.0
    looks = []
    if config.max_looks:
        for look in range(1, config.max_looks + 1):
            n_so_far = int(len(df) * look / config.max_looks)
            look_result = analyze_experiment(df.iloc[:n_so_far], config, look_number=look)
            looks.append(
                {
                    "label": f"Look {look}/{config.max_looks}",
                    "users": n_so_far,
                    "recommendation": look_result.sequential.recommendation,
                    "message": look_result.sequential.message,
                }
            )
            if look_result.sequential.recommendation != "continue":
                break

    return {
        "name": name,
        "story": story,
        "users": int(len(df)),
        "rates": rates,
        "lift_pct": float(lift_pct),
        "p_value": float(result.primary.p_value),
        "summary": result.summary(),
        "looks": looks,
        "badge": badge,
        "badge_text": badge_text,
        "side_title": side_title,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML


@app.get("/demo/scenarios")
def demo_scenarios():
    clear_df = _demo_users("demo-ui-clear", 20_000, 0.10, 0.15, seed=7)
    no_diff_df = _demo_users("demo-ui-no-diff", 20_000, 0.10, 0.10, seed=42)
    guardrail_df = _demo_users("demo-ui-guardrail", 20_000, 0.10, 0.15, seed=99, include_guardrail=True)

    return {
        "scenarios": [
            _scenario_payload(
                name="Treatment wins",
                story="Control is near 10%, treatment is near 15%.",
                df=clear_df,
                config=ExperimentConfig(
                    experiment_id="demo-ui-clear",
                    primary_metric_column="converted",
                    metric_type="proportion",
                    baseline_rate=0.10,
                    minimum_detectable_effect=0.05,
                    max_looks=5,
                ),
                badge="winner",
                badge_text="Safe to stop",
                side_title="Sequential Looks",
            ),
            _scenario_payload(
                name="No fake winner",
                story="Both variants are truly near 10%.",
                df=no_diff_df,
                config=ExperimentConfig(
                    experiment_id="demo-ui-no-diff",
                    primary_metric_column="converted",
                    metric_type="proportion",
                    max_looks=5,
                ),
                badge="no_winner",
                badge_text="No winner",
                side_title="Final Look",
            ),
            _scenario_payload(
                name="Guardrail catches risk",
                story="Conversion rises, but session duration drops.",
                df=guardrail_df,
                config=ExperimentConfig(
                    experiment_id="demo-ui-guardrail",
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
                ),
                badge="guardrail",
                badge_text="Regression",
                side_title="Guardrail Verdict",
            ),
        ]
    }


@app.post("/experiments", dependencies=[Depends(verify_api_key)])
def create_experiment(payload: ExperimentIn):
    exp_id = uuid.uuid4().hex[:12]
    config = ExperimentConfig(
        experiment_id=exp_id,
        primary_metric_column=payload.primary_metric_column,
        metric_type=payload.metric_type,
        alpha=payload.alpha,
        expected_split=payload.split,
        baseline_rate=payload.baseline_rate,
        minimum_detectable_effect=payload.minimum_detectable_effect,
        guardrails=[GuardrailConfig(**g.model_dump()) for g in payload.guardrails],
        max_looks=payload.max_looks,
    )
    experiment_key = storage.create_experiment(exp_id, payload.name, config)
    return {"experiment_id": exp_id, "name": payload.name, "experiment_key": experiment_key}


@app.post("/experiments/{experiment_id}/extend", dependencies=[Depends(verify_experiment_access)])
def extend_max_looks(experiment_id: str, payload: ExtendMaxLooksIn):
    """Raise the look budget on a still-running experiment.

    Without this, a team that needs to check results more times than
    originally planned (a launch delayed by a holiday, a slower-than-expected
    ramp-up) has no option but to hit `stop_no_effect` at the last look even
    though the experiment hasn't actually run long enough -- forcing a
    premature call. This lets them raise the ceiling instead.

    Caveat: the O'Brien-Fleming boundary is calibrated against the
    max_looks known when the sequence started. Raising it mid-flight is a
    pragmatic trade-off, not a formally re-derived alpha-spending schedule --
    prefer setting a generous max_looks upfront when possible, and treat
    this as an escape hatch, not routine practice.
    """
    row = storage.get_experiment(experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if row["locked"]:
        raise HTTPException(status_code=409, detail="Experiment is already locked; max_looks can no longer be changed.")

    config = _config_from_row(row)
    if config.max_looks is None:
        raise HTTPException(status_code=400, detail="This experiment does not use sequential testing (max_looks is null).")
    if payload.max_looks <= config.max_looks:
        raise HTTPException(status_code=400, detail=f"New max_looks ({payload.max_looks}) must be greater than the current value ({config.max_looks}).")
    if payload.max_looks <= row["look_count"]:
        raise HTTPException(status_code=400, detail=f"New max_looks ({payload.max_looks}) must be greater than looks already spent ({row['look_count']}).")

    storage.update_max_looks(experiment_id, payload.max_looks)
    return {"experiment_id": experiment_id, "max_looks": payload.max_looks}


@app.post("/experiments/{experiment_id}/events", dependencies=[Depends(verify_experiment_access)])
def log_event(experiment_id: str, payload: EventIn):
    row = storage.get_experiment(experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if row["locked"]:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is locked (sequential test recommended '{row['locked_reason']}'); no more events are accepted.",
        )

    config = _config_from_row(row)
    group = payload.group or assign_group(payload.user_id, experiment_id, config.expected_split)
    storage.upsert_event(experiment_id, payload.user_id, group, payload.metrics)
    return {"user_id": payload.user_id, "group": group}


def _build_results(result, n_users: int) -> dict:
    return {
        "status": "ok",
        "n_users": n_users,
        "srm": {
            "is_mismatched": result.srm.is_mismatched,
            "observed": result.srm.observed,
            "expected": result.srm.expected,
            "p_value": result.srm.p_value,
        },
        "primary_metric": {
            "p_value": result.primary.p_value,
            "diff": result.primary.diff,
            "relative_diff_pct": result.primary.relative_diff_pct,
            "ci": [result.primary.ci_low, result.primary.ci_high],
            "is_significant": result.primary.is_significant,
            "conclusion": result.primary.conclusion,
        },
        "sequential": (
            None
            if result.sequential is None
            else {
                "look_number": result.sequential.look_number,
                "max_looks": result.sequential.max_looks,
                "recommendation": result.sequential.recommendation,
                "message": result.sequential.message,
            }
        ),
        "sample_size_check": (
            None
            if result.sample_size_check is None
            else {
                "is_adequate": result.sample_size_check.is_adequate,
                "message": result.sample_size_check.message,
            }
        ),
        "guardrails": {
            name: {
                "regression_detected": g["regression_detected"],
                "conclusion": g["test"].conclusion,
            }
            for name, g in result.guardrails.items()
        },
        "summary": result.summary(),
    }


@app.get("/experiments/{experiment_id}/results", dependencies=[Depends(verify_experiment_access)])
def get_results(experiment_id: str):
    row = storage.get_experiment(experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if row["locked"]:
        final_result = json.loads(row["final_result_json"])
        final_result["locked"] = True
        final_result["locked_reason"] = row["locked_reason"]
        return final_result

    config = _config_from_row(row)
    events = storage.get_events(experiment_id)
    if not events:
        return {"status": "no_data", "message": "No events logged yet."}

    user_df = pd.DataFrame(events)

    required_cols = {config.primary_metric_column, *[g.column for g in config.guardrails]}
    missing = required_cols - set(user_df.columns)
    if missing:
        return {"status": "incomplete_data", "message": f"Missing metric columns so far: {missing}"}

    # A statistical test needs both groups populated -- early in a live
    # experiment (or right after creating it) one group may have zero events
    # so far, which would otherwise crash the test with a division by zero.
    empty_groups = [g for g in config.expected_split if (user_df.group == g).sum() == 0]
    if empty_groups:
        return {
            "status": "insufficient_data",
            "message": f"Nhóm {empty_groups} chưa có dữ liệu -- cần đợi thêm event trước khi phân tích.",
        }

    # Only count this as a "look" once we're actually running the test --
    # no_data/incomplete_data/insufficient_data responses above don't spend
    # any of the sequential testing boundary's alpha budget.
    look_number = storage.increment_look_count(experiment_id)
    result = analyze_experiment(user_df, config, look_number=look_number)
    response = _build_results(result, n_users=len(user_df))

    if result.sequential is not None and result.sequential.recommendation != "continue":
        response["locked"] = True
        response["locked_reason"] = result.sequential.recommendation
        storage.lock_experiment(experiment_id, result.sequential.recommendation, json.dumps(response))
    else:
        response["locked"] = False

    return response
