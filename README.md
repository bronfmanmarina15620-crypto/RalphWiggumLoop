# SMAPS — Self-learning Market Analysis & Prediction System

Daily-only MVP: predict next-day stock direction for a watchlist of tickers,
evaluate predictions against realized outcomes, and retrain on degradation.
No execution, no intraday signals.

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Verification (Scaffold Health)

Run the full scaffold verification in one shot:

```bash
bash scripts/verify_scaffold.sh
```

Success criteria: all six steps pass and the working tree is clean.

The script runs, in order:

1. `pip install -e ".[dev]"`
2. `python -m pytest -q`
3. `python -c "import smaps; print('ok')"`
4. `make test`
5. `make smoke`
6. `git status --porcelain` (must be empty)
