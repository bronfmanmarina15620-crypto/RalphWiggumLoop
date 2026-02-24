# PRD: SMAPS — Self-learning Market Analysis & Prediction System

## Scope

Daily-only MVP: predict next-day stock direction (UP/DOWN/FLAT) for a watchlist of
tickers, evaluate predictions against realized outcomes, and retrain automatically on
degradation. No execution, no intraday signals.

---

## Phase 0 — Repo / Infra Skeleton

- [x] **US-0001**: Create PRD.md + progress.txt (scaffold)
  - [x] `PRD.md` exists at repo root with all phases and user stories listed
  - [x] `progress.txt` exists at repo root with sections: Learnings, Decisions, Integration Notes, Next Smallest Task
  - [x] Both files are committed and trackable by RALPH
  - Verified by: `bash scripts/verify_scaffold.sh`

- [x] **US-0002**: Add Python project skeleton (src/, tests/, pyproject.toml)
  - [x] `pyproject.toml` exists with project name `smaps`, Python >=3.11
  - [x] `src/smaps/__init__.py` exists and exposes `__version__`
  - [x] `tests/` directory exists with at least one test file
  - [x] `pip install -e .` succeeds without errors
  - [x] `python -c "import smaps; print(smaps.__version__)"` prints a version string
  - Verified by: `bash scripts/verify_scaffold.sh`

- [x] **US-0003**: Add `make test` / `pytest -q` + one smoke test
  - [x] `Makefile` exists with `test` and `smoke` targets
  - [x] `make test` runs `python -m pytest -q` and exits 0
  - [x] `tests/test_smoke.py` contains at least 2 passing tests (import + DB schema)
  - [x] `make smoke` runs `python -c "import smaps; print('ok')"` and exits 0
  - Verified by: `bash scripts/verify_scaffold.sh`

- [x] **US-0004**: Add `.env.example` + config loader
  - [x] `.env.example` exists with placeholder keys: `SMAPS_TICKERS`, `SMAPS_DB_PATH`, `SMAPS_LOG_LEVEL`
  - [x] `src/smaps/config.py` defines a `Settings` class with typed fields and defaults
  - [x] `Settings()` instantiation succeeds with no env vars set (all defaults work)
  - [x] Environment variable overrides are respected (e.g., `SMAPS_LOG_LEVEL=DEBUG`)
  - Verified by: `bash scripts/verify_scaffold.sh`

- [x] **US-0005**: Add logging baseline + structured run_id
  - [x] `src/smaps/logging.py` exports `get_logger(name, run_id=None)`
  - [x] Logger output includes ISO timestamp, level, name, and run_id when provided
  - [x] `get_logger("test")` returns a usable `logging.Logger` instance
  - [x] No external logging dependencies required
  - Verified by: `bash scripts/verify_scaffold.sh`

- [ ] **US-0006**: Add local SQLite schema migration tool
  - [ ] `src/smaps/db.py` exports `get_connection(db_path)` and `ensure_schema(conn)`
  - [ ] `ensure_schema` creates `ohlcv_daily` table with columns: ticker, date, open, high, low, close, adj_close, volume
  - [ ] Primary key is `(ticker, date)`
  - [ ] `ensure_schema` is idempotent (safe to call multiple times)
  - [ ] Unit test creates in-memory DB, calls `ensure_schema`, verifies table exists

---

## Phase 1 — DataCollector (Daily only, V1)

- [ ] **US-0101**: Define canonical data model for OHLCV daily bars
  - [ ] Pydantic model (or dataclass) `OHLCVBar` defined in `src/smaps/models.py`
  - [ ] Fields: ticker (str), date (date), open (float), high (float), low (float), close (float), adj_close (float), volume (float)
  - [ ] Validation: high >= low, volume >= 0
  - [ ] Unit test verifies model creation and validation

- [ ] **US-0102**: Implement Yahoo Finance daily downloader for 1 ticker
  - [ ] `src/smaps/collector.py` exports `fetch_daily_bars(ticker, start, end) -> list[OHLCVBar]`
  - [ ] Function downloads OHLCV data via yfinance and returns validated bars
  - [ ] Bars are inserted into `ohlcv_daily` table
  - [ ] Unit test with mocked yfinance response verifies parsing

- [ ] **US-0103**: Add adjusted prices handling (splits/dividends)
  - [ ] Downloader uses `auto_adjust=True` or equivalent to handle corporate actions
  - [ ] `adj_close` column is populated correctly
  - [ ] Integration test confirms adjusted vs raw prices differ when expected

- [ ] **US-0104**: Add data-quality checks: range/outliers/missing policy
  - [ ] Function `check_bar_quality(bar) -> list[str]` returns list of warnings
  - [ ] Checks: OHLC range sanity, zero-volume detection, missing-date gaps
  - [ ] Quality warnings are logged but do not block ingestion
  - [ ] Unit tests cover each check category

- [ ] **US-0105**: Add multi-ticker ingestion (N tickers) + idempotent upsert
  - [ ] `ingest_tickers(tickers, start, end)` processes a list of tickers
  - [ ] Uses `INSERT OR REPLACE` (upsert) to avoid duplicates
  - [ ] Errors for one ticker do not block others
  - [ ] Unit test: insert same row twice, verify single row in DB

- [ ] **US-0106**: Add provider fallback interface (stub)
  - [ ] `DataProvider` protocol/ABC defined with `fetch_daily_bars` method signature
  - [ ] `YahooProvider` implements the protocol
  - [ ] `FallbackProvider` stub accepts primary + fallback providers
  - [ ] Unit test verifies fallback is called when primary raises

---

## Phase 2 — FeatureEngineer

- [ ] **US-0201**: Feature pipeline interface: input bars -> output feature frame
  - [ ] Interface/protocol `FeaturePipeline` defined with `transform(bars) -> DataFrame` signature
  - [ ] Docstring specifies input/output contract
  - [ ] Unit test stub exists

- [ ] **US-0202**: Implement MA/EMA/MACD features
  - [ ] Functions compute MA(5,20), EMA(12,26), MACD line + signal
  - [ ] Output columns are named consistently (e.g., `ma_5`, `ema_12`, `macd`)
  - [ ] Unit test with synthetic data verifies output shape and column names

- [ ] **US-0203**: Implement RSI/CCI/Stoch features
  - [ ] Functions compute RSI(14), CCI(20), Stochastic %K/%D
  - [ ] NaN handling documented (first N rows)
  - [ ] Unit test verifies output range (RSI 0-100, Stoch 0-100)

- [ ] **US-0204**: Implement ATR + Bollinger features
  - [ ] Functions compute ATR(14) and Bollinger Bands (20, 2σ)
  - [ ] Unit test verifies Bollinger upper > middle > lower

- [ ] **US-0205**: Feature snapshot persistence (store feature vector per date+ticker)
  - [ ] Table `feature_snapshots` created with schema defined
  - [ ] Snapshot includes feature_date, ticker, JSON blob of feature values, pipeline_version
  - [ ] Unit test verifies round-trip: save and load snapshot

- [ ] **US-0206**: Prevent leakage (no future bars) unit tests
  - [ ] Feature computation at date T uses only bars dated <= T
  - [ ] Unit test: inject known future data, verify it does not appear in features at T
  - [ ] Documentation note on leakage prevention strategy

---

## Phase 3 — Prediction Agent v1

- [ ] **US-0301**: Define prediction target: direction (UP/DOWN/FLAT) at horizon 1d
  - [ ] Enum or literal type for direction defined
  - [ ] Threshold for FLAT defined and configurable (e.g., ±0.1%)
  - [ ] Docstring specifies labeling logic

- [ ] **US-0302**: Train baseline model (LogReg / XGBoost)
  - [ ] Training function accepts feature DataFrame + labels, returns fitted model
  - [ ] Default algorithm is LogisticRegression; XGBoost available via config flag
  - [ ] Train/test split strategy documented (time-based, no shuffle)

- [ ] **US-0303**: Calibrate probabilities (Platt/Isotonic)
  - [ ] Calibration wrapper applied post-training
  - [ ] Calibrated model exposes `predict_proba`
  - [ ] Unit test: calibrated probabilities sum to 1.0

- [ ] **US-0304**: Persist model artifacts + model_version
  - [ ] Model saved to `models/<ticker>_<version>.joblib`
  - [ ] `models/registry.json` tracks version history
  - [ ] Load function retrieves latest model for a ticker

- [ ] **US-0305**: Prediction API: {direction, confidence, horizon}
  - [ ] Function `predict(ticker, date) -> PredictionResult`
  - [ ] `PredictionResult` contains direction, confidence (0-1), horizon, model_version
  - [ ] Unit test with mock model verifies output schema

- [ ] **US-0306**: Save predictions with timestamp + feature_snapshot_id + model_version
  - [ ] Table `predictions` created with foreign keys to feature_snapshots and model registry
  - [ ] Each prediction row includes created_at timestamp
  - [ ] Unit test verifies prediction persistence

---

## Phase 4 — Evaluator

- [ ] **US-0401**: Match predictions to realized outcome
  - [ ] Function `evaluate_prediction(prediction_id) -> EvalResult`
  - [ ] Matches predicted direction to actual next-day movement
  - [ ] Handles missing market data gracefully (weekends, holidays)

- [ ] **US-0402**: Compute accuracy + per-class precision/recall
  - [ ] Metrics computed over configurable window (default 90 days)
  - [ ] Breakdown by direction class (UP/DOWN/FLAT)
  - [ ] Output serializable as JSON

- [ ] **US-0403**: Compute calibration error
  - [ ] Expected Calibration Error (ECE) computed with configurable bin count
  - [ ] Docstring explains ECE formula and interpretation

- [ ] **US-0404**: Rolling 90D window report serialization
  - [ ] Report dataclass with accuracy, precision, recall, ECE, window dates
  - [ ] Serialized to `reports/eval_<date>.json`
  - [ ] Unit test verifies report schema

---

## Phase 5 — MetaLearner + RiskGuard

- [ ] **US-0501**: Retrain trigger rule (performance degradation threshold)
  - [ ] Configurable accuracy threshold (default: drop below 50% over 30D)
  - [ ] Trigger emits structured log event
  - [ ] Docstring specifies trigger logic

- [ ] **US-0502**: Optuna HPO (small search space)
  - [ ] Optuna study with ≤10 trials for regularization + feature subset
  - [ ] Best params logged and persisted
  - [ ] Interface defined; implementation can be stubbed initially

- [ ] **US-0503**: OOS validation gate before deploying model_version
  - [ ] Hold-out OOS period (e.g., last 30 days) used for validation
  - [ ] Model only promoted if OOS accuracy > threshold
  - [ ] Gate decision logged

- [ ] **US-0504**: Rollback rule if new model underperforms
  - [ ] Compare new model vs current on OOS data
  - [ ] Rollback to previous version if new is worse
  - [ ] Rollback event logged with reason

- [ ] **US-0505**: Drift detection on features (KS-test)
  - [ ] KS-test applied to each feature column: training distribution vs recent window
  - [ ] Alert if p-value < configurable threshold (default 0.05)
  - [ ] Drift report persisted

- [ ] **US-0506**: Overfit guard (train/val gap gate + circuit breaker)
  - [ ] Train/val accuracy gap threshold defined (e.g., >15% gap = overfit)
  - [ ] Circuit breaker halts deployment if gap exceeded
  - [ ] Docstring explains gap computation

---

## Phase 6 — Orchestrator + Scheduling

- [ ] **US-0601**: Orchestrator run: collect -> features -> predict
  - [ ] `run_pipeline(tickers, date)` chains collector, feature engineer, predictor
  - [ ] Each step logged with timing
  - [ ] Failure in one step stops pipeline for that ticker, continues others

- [ ] **US-0602**: Daily schedule (GitHub Actions)
  - [ ] `.github/workflows/daily.yml` runs pipeline on cron (e.g., 22:00 UTC weekdays)
  - [ ] Workflow installs deps, runs pipeline, commits results
  - [ ] Manual trigger (`workflow_dispatch`) supported

- [ ] **US-0603**: Evaluator schedule (after close / next day)
  - [ ] Separate workflow or step runs evaluator after market data available
  - [ ] Evaluator results persisted before next prediction cycle
  - [ ] Schedule documented

- [ ] **US-0604**: Message bus (V1: DB queue)
  - [ ] Table `events` with columns: id, event_type, payload (JSON), created_at, processed_at
  - [ ] Producer writes events; consumer reads unprocessed events
  - [ ] Interface designed for future Kafka migration

---

## Phase 7 — Reporting / API

- [ ] **US-0701**: FastAPI endpoint: latest predictions per ticker
  - [ ] `GET /predictions/latest` returns JSON array of latest predictions
  - [ ] Filterable by ticker query parameter
  - [ ] Response schema documented

- [ ] **US-0702**: Endpoint: performance summary (90D)
  - [ ] `GET /performance` returns accuracy, precision, recall, ECE for last 90 days
  - [ ] Filterable by ticker
  - [ ] Response includes window start/end dates

- [ ] **US-0703**: Minimal dashboard (static HTML) showing metrics + latest signals
  - [ ] HTML page generated from latest report + predictions
  - [ ] Shows: prediction table, accuracy chart, model version info
  - [ ] Serveable as static file or via FastAPI

---

## Non-Goals

- Live trading execution or order placement
- Intraday signals or sub-daily predictions
- Guaranteed profitability or financial advice
- Multi-asset-class support (equities only for MVP)
