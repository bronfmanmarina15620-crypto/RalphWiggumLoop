# PRD: SMAPS — Self-learning Market Analysis & Prediction System

## Introduction

SMAPS is an autonomous stock prediction system that forecasts next-day price direction (UP or DOWN) for a small watchlist of 5–10 tickers. What makes it unique is its closed feedback loop: the system predicts, evaluates its accuracy against realized outcomes, and retrains its ML model automatically when performance degrades — no human intervention required.

The system ingests three data sources — historical price data (OHLCV), news/social sentiment, and fundamental indicators — to build a rich feature set for prediction.

## Goals

- Predict next-day direction (UP/DOWN) for each ticker in a configurable watchlist
- Ingest and combine price, sentiment, and fundamental data into a unified feature pipeline
- Evaluate predictions daily against actual market outcomes
- Automatically retrain the ML model when accuracy drops below a threshold
- Run the full predict → evaluate → retrain loop autonomously on a daily schedule
- Persist all predictions, evaluations, and model versions for auditability

## User Stories

---

### Phase 0 — Project Scaffold

### US-001: Create project skeleton
**Description:** As a developer, I want a Python project structure so that all future code has a consistent home.

**Acceptance Criteria:**
- [x] `pyproject.toml` exists with project name `smaps`, Python >=3.11
- [x] `src/smaps/__init__.py` exists and exposes `__version__`
- [x] `tests/` directory exists with `conftest.py`
- [x] `pip install -e .` succeeds
- [x] Typecheck passes

### US-002: Add Makefile with test and lint targets
**Description:** As a developer, I want standard make targets so that CI and local dev use the same commands.

**Acceptance Criteria:**
- [x] `Makefile` with targets: `test`, `lint`, `typecheck`
- [x] `make test` runs `python -m pytest -q` and exits 0
- [x] `make typecheck` runs `mypy src/` and exits 0
- [x] Typecheck passes

### US-003: Add config loader with environment variable support
**Description:** As a developer, I want a typed config so that all settings are centralized and overridable.

**Acceptance Criteria:**
- [x] `.env.example` with keys: `SMAPS_TICKERS`, `SMAPS_DB_PATH`, `SMAPS_LOG_LEVEL`
- [x] `src/smaps/config.py` defines `Settings` dataclass with typed fields and defaults
- [x] `Settings()` works with no env vars (all defaults)
- [x] Env var overrides are respected
- [x] Typecheck passes

### US-004: Add structured logging with run_id
**Description:** As a developer, I want structured logs so that each pipeline run is traceable.

**Acceptance Criteria:**
- [x] `src/smaps/logging.py` exports `get_logger(name, run_id=None)`
- [x] Output includes ISO timestamp, level, name, and run_id
- [x] Unit test verifies logger returns `logging.Logger`
- [x] Typecheck passes

### US-005: Add SQLite database layer with schema migrations
**Description:** As a developer, I want a local DB with versioned migrations so that schema evolves safely.

**Acceptance Criteria:**
- [x] `src/smaps/db.py` exports `get_connection(db_path)` and `ensure_schema(conn)`
- [x] `schema_migrations` table tracks applied version
- [x] Migrations are sequential and idempotent
- [x] Unit test: fresh DB → ensure_schema → verify tables exist
- [x] Typecheck passes

---

### Phase 1 — Data Collection

### US-101: Define OHLCV data model
**Description:** As a developer, I want a canonical bar model so that all data flows use a consistent structure.

**Acceptance Criteria:**
- [x] Dataclass `OHLCVBar` in `src/smaps/models.py` with fields: ticker, date, open, high, low, close, volume
- [x] Validation: high >= low, volume >= 0
- [x] Unit test verifies creation and validation
- [x] Typecheck passes

### US-102: Implement Yahoo Finance daily downloader
**Description:** As a developer, I want to fetch daily OHLCV bars so that the system has price data.

**Acceptance Criteria:**
- [x] `src/smaps/collectors/price.py` exports `fetch_daily_bars(ticker, start, end) -> list[OHLCVBar]`
- [x] Uses yfinance with `auto_adjust=True`
- [x] Unit test with mocked yfinance verifies parsing
- [x] Typecheck passes

### US-103: Add OHLCV table and idempotent upsert
**Description:** As a developer, I want price bars persisted so that features can be computed offline.

**Acceptance Criteria:**
- [x] Migration adds `ohlcv_daily` table: ticker, date, open, high, low, close, volume; PK (ticker, date)
- [x] `INSERT OR REPLACE` upsert avoids duplicates
- [x] Unit test: insert same row twice → single row in DB
- [x] Typecheck passes

### US-104: Implement sentiment data collector
**Description:** As a developer, I want to ingest daily sentiment scores so that predictions include market mood.

**Acceptance Criteria:**
- [x] `src/smaps/collectors/sentiment.py` exports `fetch_sentiment(ticker, date) -> SentimentScore`
- [x] `SentimentScore` dataclass: ticker, date, score (float -1..1), source (str)
- [x] Provider uses a free news API or RSS-based heuristic
- [x] Unit test with mocked response verifies parsing
- [x] Typecheck passes

### US-105: Add sentiment table and persistence
**Description:** As a developer, I want sentiment scores stored so that features can use them.

**Acceptance Criteria:**
- [x] Migration adds `sentiment_daily` table: ticker, date, score, source; PK (ticker, date, source)
- [x] Upsert logic avoids duplicates
- [x] Unit test verifies round-trip persistence
- [x] Typecheck passes

### US-106: Implement fundamentals data collector
**Description:** As a developer, I want to ingest key fundamental metrics so that predictions include valuation context.

**Acceptance Criteria:**
- [x] `src/smaps/collectors/fundamentals.py` exports `fetch_fundamentals(ticker) -> Fundamentals`
- [x] `Fundamentals` dataclass: ticker, date, pe_ratio, market_cap, eps, revenue (all optional floats)
- [x] Uses yfinance `.info` or equivalent free source
- [x] Unit test with mocked response verifies parsing
- [x] Typecheck passes

### US-107: Add fundamentals table and persistence
**Description:** As a developer, I want fundamentals stored so that features can reference them.

**Acceptance Criteria:**
- [x] Migration adds `fundamentals_daily` table: ticker, date, pe_ratio, market_cap, eps, revenue
- [x] PK (ticker, date); upsert logic
- [x] Unit test verifies round-trip persistence
- [x] Typecheck passes

### US-108: Add multi-ticker ingestion orchestrator
**Description:** As a developer, I want all tickers ingested in one call so that the pipeline is simple.

**Acceptance Criteria:**
- [x] `src/smaps/collectors/ingest.py` exports `ingest_all(tickers, start, end)`
- [x] Calls price, sentiment, and fundamentals collectors for each ticker
- [x] Error in one ticker does not block others
- [x] Each step logged with timing
- [x] Typecheck passes

---

### Phase 2 — Feature Engineering

### US-201: Define feature pipeline interface
**Description:** As a developer, I want a standard interface so that feature pipelines are composable.

**Acceptance Criteria:**
- [x] Protocol `FeaturePipeline` with `transform(ticker, as_of_date) -> dict[str, float]`
- [x] Docstring specifies no-future-data contract
- [x] Typecheck passes

### US-202: Implement technical indicator features
**Description:** As a developer, I want price-derived features so that the model captures momentum and volatility patterns.

**Acceptance Criteria:**
- [x] Computes: return_1d, return_5d, return_10d, MA(5)/MA(20) ratio, volume_change_1d, volatility_20d, RSI(14)
- [x] Uses only bars dated <= as_of_date (no leakage)
- [x] Unit test with synthetic data verifies output keys and shape
- [x] Typecheck passes

### US-203: Implement sentiment features
**Description:** As a developer, I want sentiment-derived features so that the model captures market mood.

**Acceptance Criteria:**
- [x] Computes: latest_sentiment_score, sentiment_ma_5d (5-day rolling average)
- [x] Gracefully returns 0.0 if no sentiment data available
- [x] Unit test verifies output
- [x] Typecheck passes

### US-204: Implement fundamental features
**Description:** As a developer, I want fundamental-derived features so that the model captures valuation context.

**Acceptance Criteria:**
- [x] Computes: pe_ratio, eps, market_cap (latest available values)
- [x] Gracefully returns None/NaN for missing fields
- [x] Unit test verifies output
- [x] Typecheck passes

### US-205: Combine all features into unified vector
**Description:** As a developer, I want one function that returns the full feature vector for a ticker+date.

**Acceptance Criteria:**
- [x] `build_features(ticker, as_of_date) -> dict[str, float]` merges technical + sentiment + fundamental features
- [x] Returns consistent key set regardless of data availability
- [x] Unit test verifies combined output
- [x] Typecheck passes

### US-206: Add feature snapshot persistence
**Description:** As a developer, I want feature vectors stored so that predictions are reproducible.

**Acceptance Criteria:**
- [x] Migration adds `feature_snapshots` table: id, ticker, feature_date, features_json, pipeline_version
- [x] Round-trip test: save and load snapshot, verify equality
- [x] Typecheck passes

### US-207: Add leakage prevention tests
**Description:** As a developer, I want proof that no future data leaks into features.

**Acceptance Criteria:**
- [x] Test: inject known future bar, verify it does not appear in features at date T
- [x] Test: feature_date in snapshot <= as_of_date
- [x] Typecheck passes

---

### Phase 3 — Prediction Model

### US-301: Define prediction result model
**Description:** As a developer, I want a canonical prediction type so that downstream code has a stable contract.

**Acceptance Criteria:**
- [x] Dataclass `PredictionResult`: ticker, prediction_date, direction (UP/DOWN), confidence (0-1), model_version
- [x] Unit test verifies creation
- [x] Typecheck passes

### US-302: Train baseline model (Logistic Regression)
**Description:** As a developer, I want a trainable model so that the system can make predictions.

**Acceptance Criteria:**
- [x] `src/smaps/model/trainer.py` exports `train_model(features_df, labels) -> TrainedModel`
- [x] Uses time-based train/test split (no shuffle)
- [x] Default: LogisticRegression with StandardScaler
- [x] Unit test: train on synthetic data, verify model produces predictions
- [x] Typecheck passes

### US-303: Persist model artifacts with versioning
**Description:** As a developer, I want models saved and versioned so that I can track which model made which prediction.

**Acceptance Criteria:**
- [x] Model saved to `models/<ticker>_v<N>.joblib`
- [x] Migration adds `model_registry` table: id, ticker, version, trained_at, metrics_json, artifact_path
- [x] `load_latest_model(ticker)` retrieves the most recent model
- [x] Unit test verifies save/load round-trip
- [x] Typecheck passes

### US-304: Implement daily prediction function
**Description:** As a developer, I want one function that takes a ticker and date and returns a prediction.

**Acceptance Criteria:**
- [x] `predict(ticker, date) -> PredictionResult` loads model, builds features, returns prediction
- [x] Falls back to error if no trained model exists
- [x] Unit test with mock model verifies output schema
- [x] Typecheck passes

### US-305: Persist predictions to database
**Description:** As a developer, I want predictions stored so that evaluation can match them to outcomes.

**Acceptance Criteria:**
- [x] Migration adds `predictions` table: id, ticker, prediction_date, direction, confidence, model_version, feature_snapshot_id, created_at
- [x] Unit test verifies prediction round-trip
- [x] Typecheck passes

---

### Phase 4 — Evaluator

### US-401: Match predictions to realized outcomes
**Description:** As a developer, I want each prediction scored against actual price movement so that accuracy is measurable.

**Acceptance Criteria:**
- [x] `evaluate_prediction(prediction_id) -> EvalResult` compares predicted vs actual direction
- [x] Handles weekends/holidays (skips non-trading days)
- [x] `EvalResult`: prediction_id, actual_direction, is_correct (bool), evaluated_at
- [x] Unit test verifies correct/incorrect classification
- [x] Typecheck passes

### US-402: Compute rolling accuracy metrics
**Description:** As a developer, I want aggregate accuracy stats so that model health is monitorable.

**Acceptance Criteria:**
- [x] `compute_metrics(ticker, window_days=90) -> MetricsReport`
- [x] Includes: accuracy, precision, recall (per UP/DOWN class), total predictions
- [x] Output serializable as JSON
- [x] Unit test with known outcomes verifies metric calculation
- [x] Typecheck passes

### US-403: Persist evaluation results
**Description:** As a developer, I want evaluation results stored so that the retrain trigger can query them.

**Acceptance Criteria:**
- [x] Migration adds `evaluations` table: id, prediction_id, actual_direction, is_correct, evaluated_at
- [x] `reports/` directory stores JSON metric reports
- [x] Unit test verifies persistence
- [x] Typecheck passes

---

### Phase 5 — Self-Learning Loop

### US-501: Implement retrain trigger based on accuracy degradation
**Description:** As a developer, I want the system to detect when it's underperforming so that retraining happens automatically.

**Acceptance Criteria:**
- [x] `should_retrain(ticker, threshold=0.50, window_days=30) -> bool`
- [x] Returns True when rolling accuracy drops below threshold
- [x] Emits structured log event on trigger
- [x] Unit test with synthetic eval results verifies trigger logic
- [x] Typecheck passes

### US-502: Implement automated retraining pipeline
**Description:** As a developer, I want retraining to happen end-to-end so that no human intervention is needed.

**Acceptance Criteria:**
- [x] `retrain(ticker)` fetches latest data, builds features, trains new model, saves with incremented version
- [x] Uses all available historical data (not just recent window)
- [x] Logs new model version and training metrics
- [x] Unit test verifies new model version is created
- [x] Typecheck passes

### US-503: Add out-of-sample validation gate
**Description:** As a developer, I want a safety check before deploying a new model so that bad models don't go live.

**Acceptance Criteria:**
- [x] Hold-out OOS period (last 30 days) used for validation
- [x] New model only promoted if OOS accuracy > current model accuracy
- [x] Gate decision logged with metrics
- [x] Unit test verifies gate blocks inferior model
- [x] Typecheck passes

### US-504: Add rollback on regression
**Description:** As a developer, I want automatic rollback so that a bad retrain doesn't degrade the system.

**Acceptance Criteria:**
- [x] If new model fails OOS gate, previous model version remains active
- [x] Rollback event logged with reason and metrics
- [x] Unit test verifies rollback keeps previous model active
- [x] Typecheck passes

### US-505: Add feature drift detection
**Description:** As a developer, I want drift alerts so that the system knows when its inputs are changing.

**Acceptance Criteria:**
- [x] KS-test on each feature: training distribution vs recent 30-day window
- [x] Alert logged if p-value < 0.05
- [x] Drift report persisted to `reports/drift_<date>.json`
- [x] Unit test with shifted distribution verifies detection
- [x] Typecheck passes

---

### Phase 6 — Orchestrator & Scheduling

### US-601: Implement daily pipeline orchestrator
**Description:** As a developer, I want one entry point that runs the full daily cycle.

**Acceptance Criteria:**
- [x] `run_pipeline(tickers, date)` chains: ingest → features → predict → evaluate → retrain-if-needed
- [x] Each step logged with timing and run_id
- [x] Failure in one ticker does not block others
- [x] Unit test with mocked components verifies call sequence
- [x] Typecheck passes

### US-602: Add GitHub Actions daily schedule
**Description:** As a developer, I want the pipeline to run automatically every trading day.

**Acceptance Criteria:**
- [x] `.github/workflows/daily.yml` runs pipeline on cron (weekdays 22:00 UTC)
- [x] Installs dependencies, runs `python -m smaps.pipeline`
- [x] Manual trigger via `workflow_dispatch` supported
- [x] Typecheck passes

### US-603: Add CLI entry point
**Description:** As a developer, I want a CLI so that I can run the pipeline manually.

**Acceptance Criteria:**
- [x] `python -m smaps.pipeline --tickers AAPL,MSFT --date 2025-01-15` runs full pipeline
- [x] `--dry-run` flag logs steps without executing
- [x] `--help` shows usage
- [x] Typecheck passes

---

### Phase 7 — Reporting & Dashboard

### US-701: Add FastAPI endpoint for latest predictions
**Description:** As a user, I want to see today's predictions so that I know the system's current signals.

**Acceptance Criteria:**
- [x] `GET /predictions/latest` returns JSON array of latest predictions
- [x] Filterable by `?ticker=AAPL`
- [x] Response includes direction, confidence, model_version, prediction_date
- [x] Typecheck passes
- [x] Verify changes work in browser

### US-702: Add performance summary endpoint
**Description:** As a user, I want to see how accurate the system is so that I can gauge trust.

**Acceptance Criteria:**
- [x] `GET /performance` returns 90-day accuracy, precision, recall per ticker
- [x] Response includes window start/end dates
- [x] Typecheck passes
- [x] Verify changes work in browser

### US-703: Add minimal HTML dashboard
**Description:** As a user, I want a visual dashboard so that I can monitor predictions and accuracy at a glance.

**Acceptance Criteria:**
- [x] Static HTML page showing: prediction table, accuracy chart, last retrain date
- [x] Served via FastAPI at `/dashboard`
- [x] Auto-refreshes data on load
- [x] Typecheck passes
- [x] Verify changes work in browser

---

## Non-Goals

- **No live trading**: The system predicts direction only; it does not place orders or manage a portfolio
- **No intraday signals**: Predictions are daily horizon only; no sub-day or real-time streaming
- **No financial advice**: Outputs are experimental; the system carries no guarantees of profitability
- **No multi-asset**: Equities only for MVP; no crypto, forex, options, or commodities
- **No paid data feeds**: MVP uses free data sources only (yfinance, free news APIs)

## Technical Notes

- **Language:** Python 3.11+
- **Database:** SQLite (local, file-based; no external DB dependency)
- **ML:** scikit-learn (LogisticRegression baseline); future stories may add XGBoost
- **Data sources:** yfinance (OHLCV + fundamentals), free news/RSS for sentiment
- **Scheduling:** GitHub Actions cron for MVP; can migrate to APScheduler or Celery later
- **Testing:** pytest with mocked external APIs; no live API calls in tests