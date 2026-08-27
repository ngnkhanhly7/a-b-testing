# A/B Testing Platform

## Business value

Uncontrolled peeking raises the false-positive rate from a nominal 5% to
17.5% (measured in checkpoint 4) — meaning that out of roughly every 6
experiments a team concludes are "statistically significant" by peeking
every day, as many as 1 may actually have no real effect at all, leading to
shipping a change that adds no value (or worse, is harmful) based on false
"evidence". The cost of shipping wrong like that — engineering effort,
roadmap time, and risk to guardrail metrics (e.g. retention, revenue) — is
usually far greater than the cost of building a platform that prevents this
by design instead of relying on an analyst's personal discipline.

## Problem

Many teams run A/B tests by hand and fall into common statistical traps:
peeking (checking results every day and stopping as soon as they look
"significant"), concluding before the sample size is adequate, and A/B
splits that are skewed by a system bug that no one checks for. This
platform lets any team define an experiment, automatically randomize users
(deterministic hashing), collect event logs, and get back a trustworthy
statistical verdict — with the primary goal of **preventing those mistakes
by design**, not by relying on the analyst's personal discipline.

## Architecture

```
src/
├── simulator.py       # generates synthetic traffic with known ground truth (the most important checkpoint)
├── assignment.py       # A/B group assignment via consistent hashing + Sample Ratio Mismatch check
├── stats_engine.py     # two-proportion z-test / Welch's t-test, CI, effect size
├── sequential.py       # proves and fixes the peeking problem with an O'Brien-Fleming boundary
├── power.py             # computes required sample size, warns when the sample size is inadequate
├── pipeline.py           # wires it together: assign (hashing) -> SRM check -> primary metric (gated by sequential testing if enabled) -> sample size -> guardrail metrics
└── api/main.py           # FastAPI: POST /experiments, POST /experiments/{id}/events, GET /experiments/{id}/results (every call counts as one "look" and applies sequential testing automatically), POST /experiments/{id}/extend (raises max_looks on a still-running experiment)
```

`pipeline.py` is where the whole flow comes together: every call to
`GET /results` counts as one "peek" (`look_number`), and if the experiment
is configured with `max_looks`, the primary metric's verdict no longer
relies on a static p<alpha threshold but is instead checked against the
O'Brien-Fleming boundary for that look — meaning a team can check
`/results` as many times as they want while the experiment is running
without inflating the false-positive rate, exactly as checkpoint 4 proves.
`max_looks=20` by default when an experiment is created via the API.

## How the tool was validated (main highlight)

This project builds a **measurement tool**, not a predictive model —
"correct" means *reaching the right statistical conclusion*. So before
trusting the tool on real data, every module is validated against simulated
data with a known answer (`src/simulator.py`), using 3 fixed scenarios:

| Scenario | p_control | p_treatment | The tool must conclude |
|---|---|---|---|
| 1. Clear difference | 10% | 15% | Statistically significant |
| 2. No difference | 10% | 10% | Not enough evidence (no false positive) |
| 3. Small difference | 10% | 10.3% | Only detectable with an adequate sample size (power analysis) |

`src/sequential.py` goes further: it **empirically measures** the
false-positive rate when peeking every day on scenario 2 (no true
difference), showing it is well above the nominal 5%, then applies the
O'Brien-Fleming boundary and measures again — results saved to
`reports/sequential_validation.md`.

### Actual validation results (already run)

| Checkpoint | Result |
|---|---|
| 3 — Stats engine | Scenario 1: p≈4.4e-23 (correctly detected). Scenario 2: p=0.86 (no false positive). Scenario 3: p=0.83 at 20k/group (not enough sample to detect — as theory predicts, see checkpoint 5). Details: `reports/stats_engine_validation.md`. |
| 4 — Peeking | Naive daily peeking: false-positive rate **17.5%** (vs. a nominal 5%). After applying O'Brien-Fleming: **6.5%**. Details: `reports/sequential_validation.md`. |
| 5 — Power analysis | Required sample size to detect 10%→10.3% (80% power): **159,059 users/group**. Empirical power over 200 simulation runs: only **32%** at 30% of the required sample size, **85%** at the exact computed sample size — matches theory. |
| 6 — Cookie Cats (real data) | Day-7 retention drops with statistical significance in the gate_40 group (p=0.0016, -0.82 pp); day-1 retention and rounds played show no significant difference. Matches known public analyses. Details: `reports/cookie_cats_analysis.md`. |
| 8 — Guardrail | Demo: the primary metric (CTR) wins +50%, but the guardrail (session duration) is detected to drop 25% with significance — the system correctly flags a "REGRESSION". |
| 7 — API + sequential end-to-end | Calling `/results` 5 times in a row through the real API path (a small 10% vs 11% effect experiment): the first 3 calls return "continue", the 4th switches to "stop_significant" -- validates the full pipeline: assign (hashing) → log event → sequential-safe conclusion, not just isolated per-module tests. |

During validation, 3 real bugs/gaps were found and fixed: (1) the first
power-analysis demo ran only a single simulation, so its conclusion could
be right or wrong by luck — fixed by running a 200-iteration Monte Carlo to
measure empirical power; (2) numpy scalar types (`numpy.bool_`,
`numpy.int64`, ...) don't serialize through FastAPI — fixed by casting to
plain Python types in `assignment.py` and `stats_engine.py`; (3)
`pipeline.py` and the API originally **did not actually use** sequential
testing (checkpoint 4) or consistent hashing (checkpoint 2) — `/results`
only ran a static z-test, meaning repeated API calls while an experiment
was running still suffered exactly the peeking bug the project was
designed to prevent. Fixed: `analyze_experiment` now takes `look_number`,
gates its verdict through `sequential.py`, and the API counts every call to
`/results` as one "look".

### Second cleanup & review pass

- **Performance:** `simulator.py` used to generate logs with a Python loop
  per user -- slow at ~0.9s/318k users, making the Monte Carlo in
  `power.py` (200 iterations) take ~4 minutes. Fully vectorized with
  numpy/pandas: now ~0.1s/318k users (~9x faster), Monte Carlo down to ~1
  minute, statistical results unchanged (verified against the old seed).
- **Real bug (SRM):** `check_srm` only compared groups present in the
  observed data -- if one group had zero users yet (very common right
  after creating an experiment), that group would silently drop out of the
  comparison instead of surfacing as a clear SRM warning, and scipy would
  even crash because observed/expected totals didn't match. Fixed: use
  `expected_split` as the source of the group list, with missing groups
  counted as 0.
- **Real bug (API 500):** as a consequence of the bug above, `GET /results`
  would crash with a `ZeroDivisionError` if a group had no events yet.
  Added a guard that returns `status: "insufficient_data"` instead of a 500.
- Pinned exact versions in `requirements.txt` to ensure reproducible
  results when the environment is set up again later.
- Added `tests/test_api.py` and extended `tests/test_assignment.py` to lock
  in the behaviors above, for a total of 16 tests.

## How to run

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Checkpoint 1: generate 3 simulated scenarios
python -m src.simulator

# Checkpoint 4: prove and fix the peeking problem (writes reports/sequential_validation.md)
python -m src.sequential

# Checkpoint 3: validate the stats engine on all 3 scenarios (writes reports/stats_engine_validation.md)
python -m src.validate_stats_engine

# Checkpoint 5: validate power analysis on the small-difference scenario
python -m src.power

# Checkpoint 6: analyze the real Cookie Cats dataset (download it first, see below)
python -m src.analyze_cookie_cats

# Checkpoint 3+7+8: run the pipeline end-to-end + guardrail metric demo
python -m src.pipeline

# Run the test suite
pytest

# Checkpoint 7: run the API (prints the API key on first startup, see "Running as an internal MVP")
uvicorn src.api.main:app --reload
```

### Deploying the demo to Vercel

Vercel will auto-detect the FastAPI app via `pyproject.toml` and the
entrypoint `src.api.main:app`. Install the Vercel CLI once on PowerShell,
then run from the project root:

```powershell
npm install -g vercel
vercel login
vercel
vercel --prod
```

After deploying, open the Vercel URL printed in the terminal. The demo page
lives at `/`, the Swagger API at `/docs`, and the health check at `/health`.

Note: SQLite on Vercel is only temporary storage for the Function
instance. The demo UI and read endpoints work fine, but if you need to
persist experiments/events long-term, swap it for PostgreSQL or a managed
database before using it for real.

## Running as an internal MVP

The API has been upgraded from a prototype to something "usable for
internal trials" (not public production-grade), with 3 parts:

1. **Persistence (SQLite):** every experiment/event is stored in
   `data/platform.db` (WAL mode + `synchronous=NORMAL` for fast writes that
   stay safe across an app crash). Restarting/redeploying the server no
   longer loses data.
2. **Auth (two-tier API keys):** the server generates one random admin API
   key on first startup (stored in the DB, unchanged across restarts,
   printed to the log for an operator to grab) -- this key is required to
   create new experiments. Each experiment, when created, is also issued
   its own `experiment_key` (returned exactly once in the response of
   `POST /experiments`), usable for the events/results/extend endpoints
   **of that experiment only**. A team holding only its own
   `experiment_key` cannot read or write another team's experiment, even
   though creating a new experiment still requires the admin key. Every
   request must carry an `X-API-Key: <key>` header; missing or wrong keys
   are rejected with 401 (a wrong or nonexistent experiment_id → 404).
3. **Auto-lock once sequential testing recommends stopping:** when
   `GET /results` gets a `stop_significant` or `stop_no_effect`
   recommendation (checkpoint 4), the experiment is automatically locked
   (`locked`): every subsequent `POST .../events` is rejected with 409, and
   every subsequent `GET .../results` always returns the same final
   verdict -- no further "peeking" is allowed once a stop decision has been
   made, in keeping with the spirit of sequential testing.
4. **Extending `max_looks` (`POST /experiments/{id}/extend`):** if a
   still-unlocked experiment needs to be checked more times than originally
   planned (a delayed launch, a slower-than-expected ramp-up), a team can
   raise `max_looks` instead of being forced into a premature
   `stop_no_effect` verdict. This is a pragmatic escape hatch, not a
   precisely re-derived alpha-spending schedule -- prefer setting a
   generous `max_looks` upfront, and only use this endpoint when truly
   needed.

Example usage (Windows PowerShell or bash):

```bash
uvicorn src.api.main:app --reload
# the terminal log will print: [A/B Testing Platform] API key ...: <key>

curl -X POST http://127.0.0.1:8000/experiments \
  -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"name":"demo","metric_type":"proportion","max_looks":20}'
```

Validated over real HTTP: (1) a missing/wrong API key → 401; (2) create an
experiment, restart the server, call `/results` again → the experiment is
still there (persistence); (3) after locking, sending another event → 409,
reading `/results` repeatedly → always returns the same locked-in result
(no new "look" is counted).

### Still missing for real production use (public-facing, multiple teams, SLA)

- `experiment_key` only scopes access by *experiment*, not by *team/user* —
  there's no concept of "which user created which experiment", no way to
  revoke a single leaked key on its own (the whole experiment has to be
  deleted/recreated), and the admin key remains a single global access
  point. Good enough to isolate data between teams sharing a server, but
  not full access control.
- Runs as a single process (SQLite doesn't suit multiple workers/pods
  writing concurrently) — scaling horizontally requires moving to
  PostgreSQL.
- No structured logging / metrics / alerting yet.
- No Dockerfile/CI-CD yet.

## Results on real data (Cookie Cats)

Run against `data/real/cookie_cats.csv` (90,189 users, gate_30=44,700 vs
gate_40=45,489). Conclusion: moving the level gate from 30 to 40 does
**not** improve retention — day-7 retention is in fact significantly lower
in the gate_40 group (p=0.0016). This matches existing public analyses of
this dataset (Kaggle/Medium), which is evidence the tool works correctly
on real data, not just on synthetic data. Full details in
`reports/cookie_cats_analysis.md`.

## Limitations & future work

- The O'Brien-Fleming boundary currently uses the classic approximate
  formula (`z_k = z_{alpha/2} * sqrt(K/k)`), not an exactly calibrated
  Lan-DeMets alpha-spending function -- good enough to illustrate the
  problem and the fix, but a production system should use a dedicated
  library (e.g. R's `sequential` package, or a full spending-function
  implementation).
- No Bayesian A/B testing or multi-armed bandit yet (automatically
  shifting traffic toward the better arm in real time) -- a natural
  extension when optimization matters more than pure statistical
  inference.
- If two teams run experiments on the same user population at the same
  time, there will be interaction effects between experiments that the
  platform doesn't yet handle — a namespace/layer mechanism is needed to
  separate traffic (like Google's overlapping experiment infrastructure).
- The API's storage used to be in-memory and lost data on restart -- this
  has since been replaced with SQLite (see "Running as an internal MVP"
  above); a managed database is still recommended before running this in
  production.
