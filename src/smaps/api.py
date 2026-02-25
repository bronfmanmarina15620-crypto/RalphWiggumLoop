"""FastAPI application exposing SMAPS predictions and performance data."""

from __future__ import annotations

import datetime
import sqlite3

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from smaps.config import Settings
from smaps.db import ensure_schema, get_last_retrain_dates, get_latest_predictions
from smaps.evaluator import compute_metrics

app = FastAPI(title="SMAPS API", version="0.1.0")


def _get_conn() -> sqlite3.Connection:
    """Return a DB connection using the configured db_path.

    Uses ``check_same_thread=False`` because FastAPI runs sync endpoints
    in a thread-pool.
    """
    settings = Settings()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    return conn


@app.get("/predictions/latest")
def predictions_latest(
    ticker: str | None = Query(default=None, description="Filter by ticker symbol"),
) -> list[dict[str, object]]:
    """Return the latest prediction for each ticker.

    Optionally filter by ``?ticker=AAPL``.
    """
    conn = _get_conn()
    try:
        records = get_latest_predictions(conn, ticker=ticker)
        return [
            {
                "ticker": r.ticker,
                "prediction_date": r.prediction_date.isoformat(),
                "direction": r.direction.value,
                "confidence": r.confidence,
                "model_version": r.model_version,
            }
            for r in records
        ]
    finally:
        conn.close()


@app.get("/performance")
def performance() -> list[dict[str, object]]:
    """Return 90-day accuracy, precision, and recall per ticker.

    Computes rolling metrics over the last 90 days for every ticker that
    has at least one prediction.  The window is anchored to the most recent
    prediction date for each ticker so that historical data is visible even
    when the server is started long after the last pipeline run.
    """
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT ticker, MAX(prediction_date) "
            "FROM predictions GROUP BY ticker ORDER BY ticker"
        )
        rows = cur.fetchall()
        results: list[dict[str, object]] = []
        for ticker, max_date_str in rows:
            as_of = datetime.date.fromisoformat(max_date_str)
            results.append(
                compute_metrics(conn, ticker, window_days=90, as_of_date=as_of).to_dict()
            )
        return results
    finally:
        conn.close()


@app.get("/retrain-info")
def retrain_info() -> dict[str, str]:
    """Return the last retrain date per ticker from the model registry."""
    conn = _get_conn()
    try:
        return get_last_retrain_dates(conn)
    finally:
        conn.close()


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMAPS - לוח בקרה</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f6fa; color: #2d3436; padding: 1.5rem; direction: rtl; }
  h1 { margin-bottom: 1rem; font-size: 1.6rem; }
  h2 { margin: 1.5rem 0 0.75rem; font-size: 1.2rem; color: #636e72; }
  table { width: 100%; border-collapse: collapse; background: #fff;
          border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { padding: 0.6rem 1rem; text-align: right; border-bottom: 1px solid #dfe6e9; }
  th { background: #2d3436; color: #fff; font-weight: 500; }
  tr:last-child td { border-bottom: none; }
  .up { color: #00b894; font-weight: 600; }
  .down { color: #d63031; font-weight: 600; }
  .chart-container { display: flex; flex-wrap: wrap; gap: 1rem; }
  .bar-group { background: #fff; border-radius: 6px; padding: 1rem;
               box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 220px; flex: 1; }
  .bar-group h3 { font-size: 1rem; margin-bottom: 0.5rem; }
  .bar { height: 22px; border-radius: 3px; margin: 4px 0; position: relative; direction: ltr; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-label { font-size: 0.8rem; color: #636e72; margin-bottom: 2px; }
  .bar-value { position: absolute; left: 6px; top: 2px; font-size: 0.75rem; color: #2d3436; }
  .accuracy-fill { background: #0984e3; }
  .precision-fill { background: #6c5ce7; }
  .recall-fill { background: #00cec9; }
  .retrain-section { margin-top: 1.5rem; }
  .retrain-item { background: #fff; padding: 0.6rem 1rem; border-radius: 6px;
                  box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 0.5rem;
                  display: inline-block; margin-left: 0.5rem; }
  .retrain-item .ticker { font-weight: 600; }
  .retrain-item .date { color: #636e72; margin-right: 0.5rem; }
  .empty { color: #b2bec3; font-style: italic; padding: 1rem; }
</style>
</head>
<body>
<h1>SMAPS - \u05dc\u05d5\u05d7 \u05d1\u05e7\u05e8\u05d4</h1>

<h2>\u05ea\u05d7\u05d6\u05d9\u05d5\u05ea \u05d0\u05d7\u05e8\u05d5\u05e0\u05d5\u05ea</h2>
<div id="predictions-table"><p class="empty">\u05d8\u05d5\u05e2\u05df...</p></div>

<h2>\u05d3\u05d9\u05d5\u05e7 (90 \u05d9\u05d5\u05dd \u05d0\u05d7\u05e8\u05d5\u05e0\u05d9\u05dd)</h2>
<div id="accuracy-chart" class="chart-container"><p class="empty">\u05d8\u05d5\u05e2\u05df...</p></div>

<h2 class="retrain-section">\u05d0\u05d9\u05de\u05d5\u05df \u05d0\u05d7\u05e8\u05d5\u05df</h2>
<div id="retrain-info"><p class="empty">\u05d8\u05d5\u05e2\u05df...</p></div>

<script>
(function() {
  function fetchJSON(url) {
    return fetch(url).then(function(r) { return r.json(); });
  }

  function renderPredictions(data) {
    var container = document.getElementById("predictions-table");
    if (!data.length) {
      container.innerHTML = '<p class="empty">\u05d0\u05d9\u05df \u05ea\u05d7\u05d6\u05d9\u05d5\u05ea \u05d6\u05de\u05d9\u05e0\u05d5\u05ea.</p>';
      return;
    }
    var html = "<table><tr><th>\u05e1\u05d9\u05de\u05d5\u05dc</th><th>\u05ea\u05d0\u05e8\u05d9\u05da</th><th>\u05db\u05d9\u05d5\u05d5\u05df</th>" +
               "<th>\u05d1\u05d9\u05d8\u05d7\u05d5\u05df</th><th>\u05de\u05d5\u05d3\u05dc</th></tr>";
    data.forEach(function(p) {
      var cls = p.direction === "UP" ? "up" : "down";
      var dir = p.direction === "UP" ? "\u05e2\u05dc\u05d9\u05d9\u05d4" : "\u05d9\u05e8\u05d9\u05d3\u05d4";
      html += "<tr><td>" + p.ticker + "</td><td>" + p.prediction_date +
              "</td><td class='" + cls + "'>" + dir +
              "</td><td>" + (p.confidence * 100).toFixed(1) + "%</td><td>" +
              p.model_version + "</td></tr>";
    });
    html += "</table>";
    container.innerHTML = html;
  }

  function renderPerformance(data) {
    var container = document.getElementById("accuracy-chart");
    if (!data.length) {
      container.innerHTML = '<p class="empty">\u05d0\u05d9\u05df \u05e0\u05ea\u05d5\u05e0\u05d9 \u05d1\u05d9\u05e6\u05d5\u05e2\u05d9\u05dd.</p>';
      return;
    }
    var html = "";
    data.forEach(function(m) {
      var acc = ((m.accuracy || 0) * 100).toFixed(1);
      var pUp = ((m.precision_up || 0) * 100).toFixed(1);
      var rUp = ((m.recall_up || 0) * 100).toFixed(1);
      html += '<div class="bar-group"><h3>' + m.ticker + '</h3>';
      html += '<div class="bar-label">\u05d3\u05d9\u05d5\u05e7</div>';
      html += '<div class="bar" style="background:#dfe6e9">' +
              '<div class="bar-fill accuracy-fill" style="width:' + acc + '%"></div>' +
              '<div class="bar-value">' + acc + '%</div></div>';
      html += '<div class="bar-label">\u05d3\u05d9\u05d5\u05e7 \u05e2\u05dc\u05d9\u05d9\u05d4 (Precision)</div>';
      html += '<div class="bar" style="background:#dfe6e9">' +
              '<div class="bar-fill precision-fill" style="width:' + pUp + '%"></div>' +
              '<div class="bar-value">' + pUp + '%</div></div>';
      html += '<div class="bar-label">\u05e8\u05d2\u05d9\u05e9\u05d5\u05ea \u05e2\u05dc\u05d9\u05d9\u05d4 (Recall)</div>';
      html += '<div class="bar" style="background:#dfe6e9">' +
              '<div class="bar-fill recall-fill" style="width:' + rUp + '%"></div>' +
              '<div class="bar-value">' + rUp + '%</div></div>';
      html += '<div class="bar-label" style="margin-top:6px;font-size:0.75rem">' +
              m.window_start + ' \u05e2\u05d3 ' + m.window_end +
              ' &middot; ' + m.evaluated_predictions + '/' + m.total_predictions +
              ' \u05d4\u05d5\u05e2\u05e8\u05db\u05d5</div></div>';
    });
    container.innerHTML = html;
  }

  function renderRetrain(data) {
    var container = document.getElementById("retrain-info");
    var tickers = Object.keys(data);
    if (!tickers.length) {
      container.innerHTML = '<p class="empty">\u05d0\u05d9\u05df \u05d4\u05d9\u05e1\u05d8\u05d5\u05e8\u05d9\u05d9\u05ea \u05d0\u05d9\u05de\u05d5\u05df.</p>';
      return;
    }
    var html = "";
    tickers.forEach(function(t) {
      var dt = data[t].replace("T", " ").substring(0, 19);
      html += '<span class="retrain-item"><span class="ticker">' + t +
              '</span><span class="date">' + dt + '</span></span>';
    });
    container.innerHTML = html;
  }

  Promise.all([
    fetchJSON("/predictions/latest"),
    fetchJSON("/performance"),
    fetchJSON("/retrain-info")
  ]).then(function(results) {
    renderPredictions(results[0]);
    renderPerformance(results[1]);
    renderRetrain(results[2]);
  });
})();
</script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Serve the minimal HTML dashboard.

    The page auto-refreshes data on load by fetching from the API endpoints
    via JavaScript.
    """
    return _DASHBOARD_HTML
