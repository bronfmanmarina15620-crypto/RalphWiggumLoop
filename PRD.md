# PRD: Self-learning Stock Movement Predictor

## Introduction

Build a system that forecasts whether a stock's price (e.g. Apple) will rise or fall the next day. The distinguishing feature is an autonomous learning component: after each prediction the system verifies its correctness and, if wrong, analyzes the error and adjusts its model without human intervention. The process runs daily in the background.

## Goals

- Generate daily up/down predictions for tracked stocks
- Automatically detect incorrect forecasts
- Learn from mistakes and update the prediction model without manual tuning
- Operate continuously as a background service

## User Stories

### US-001: Make daily stock movement predictions
**Description:** As an automated trading system, I want the predictor to output whether a given stock will go up or down tomorrow so that I can decide trades.

**Acceptance Criteria:**
- [x] Prediction routine runs once per trading day
- [x] Input: stock symbol; Output: up/down probability or label
- [x] Typecheck passes

### US-002: Validate prediction outcomes
**Description:** As the system, I want to compare each prediction to actual next-day price movement so that I know if I was correct.

**Acceptance Criteria:**
- [x] Historical price data fetched for comparison
- [x] Correctness evaluation performed automatically after market close
- [x] Typecheck passes

### US-003: Adjust model on errors
**Description:** As the self-learning component, I want the model to analyze mispredictions and update its parameters so that accuracy improves over time.

**Acceptance Criteria:**
- [x] Trigger retraining or weight update when prediction is incorrect
- [x] Learning occurs without human intervention
- [x] Model version/history recorded
- [x] Typecheck passes

### US-004: Background processing
**Description:** As an operator, I want the entire prediction and learning workflow to run automatically each day without manual start.

**Acceptance Criteria:**
- [ ] Scheduled job executes daily
- [ ] Logs record each prediction and learning step
- [ ] Typecheck passes

## Non‑Goals

- Providing a user-facing UI for manually entering data
- Supporting live intraday predictions
- Guaranteeing profitability or trading strategies

## Technical Notes

- Use existing machine learning frameworks (e.g., TensorFlow, PyTorch) for model
- Store historical prices in a database for evaluation and training data
- Schedule jobs via cron or task scheduler
- Keep models lightweight to fit within a single processing window

