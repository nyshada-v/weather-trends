# Weather Trend Analysis and Extreme Temperature Pattern Detection

A reproducible machine-learning workflow that turns quality-reviewed daily weather
records into interpretable long-term trends and a repeatable detector for unusually
hot or cold observations.

- **Dataset:** NOAA Global Historical Climatology Network Daily (GHCN-Daily).
- **Default scope:** a single station (NY City Central Park, `USW00094728`), 2000-2023.
- **Target:** 3-class label per observation - `normal`, `extreme_hot`, `extreme_cold`.
- **Models:** transparent baselines - logistic regression and random forest.
- **Interfaces:** a Streamlit dashboard, a command-line pipeline, and a Jupyter notebook.

---

## Prerequisites

- **Python 3.10+** (developed and tested on 3.14).
- **Internet access** on the first run, to download the NOAA data (cached afterwards).
- **Windows PowerShell** for the commands below. On macOS/Linux, substitute
  `source .venv/bin/activate` and `python -m streamlit run ...`.

Everything the app needs is declared in `requirements.txt`.

---

## Getting started

### 1. Open a terminal in the project folder

```powershell
cd D:\Projects\bdaCBP
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Build the data, features and models

This downloads the NOAA data, engineers features, labels extreme events, trains the
baseline models, and writes all reports and figures.

```powershell
python scripts/run_pipeline.py
```

On success it prints a summary (accuracy, macro-F1, extreme-F1, ROC-AUC) and creates:

- `data/raw/` - downloaded GHCN files
- `data/processed/` - labeled feature table
- `models/` - fitted models (joblib)
- `reports/figures/` - plots
- `reports/metrics/` - metrics (JSON), predictions and feature importances (CSV)

### 5. Launch the dashboard (UI)

Easiest - a one-click launcher that builds the data/models if missing, then starts the server:

```powershell
.\run_dashboard.ps1
```

Or run Streamlit directly:

```powershell
streamlit run app/dashboard.py
```

Open **http://localhost:8501** in your browser. Keep the terminal open; press `Ctrl+C`
to stop the server.

The dashboard has five tabs:

- **Overview** - station details, scope, accuracy/precision/recall/F1/ROC-AUC metric cards, labeling thresholds, and long-term trend.
- **Trends** - interactive daily and annual temperature charts, extreme-event frequency, and the annual trend figure.
- **Model performance** - confusion matrix, per-class metrics, ROC curves, seasonal error breakdown, alert metrics, and feature importance.
- **Predictions** - predicted extreme-probability timeline with an interactive alert-threshold slider (live precision/recall), top alerts, and false-positive review.
- **About** - methodology and the active configuration. The sidebar also has a **Run / refresh pipeline** button.

### 6. Optional - interactive notebook

```powershell
jupyter notebook notebooks/weather_extremes_walkthrough.ipynb
```

### 7. Optional - run the tests

```powershell
python -m unittest discover -s tests -v
```

### Quick start (after the first setup)

```powershell
cd D:\Projects\bdaCBP
.\.venv\Scripts\Activate.ps1
.\run_dashboard.ps1
```

---

## Common pipeline options

```powershell
python scripts/run_pipeline.py --station USW00094728 --start 2000-01-01 --end 2023-12-31
python scripts/run_pipeline.py --models logistic_regression random_forest --primary auto
python scripts/run_pipeline.py --hot-quantile 0.975 --cold-quantile 0.025
python scripts/run_pipeline.py --no-download        # reuse cached raw files
```

Permanent changes belong in `config.yaml` (see the reference below).

---

## Project layout

```
bdaCBP/
  config.yaml                     # single source of truth for the pipeline
  requirements.txt
  run_dashboard.ps1               # one-click launcher (builds if needed, starts UI)
  .streamlit/config.toml          # Streamlit settings (headless server, no onboarding prompt)
  src/weather_extremes/
    config.py                     # YAML loading + attribute access
    data_loader.py                # GHCN-Daily download + .dly parsing
    features.py                   # cleaning, calendar, rolling, lagged, anomaly features
    labeling.py                   # seasonal-percentile extreme labels
    model.py                      # baseline models, time-aware split, predictions
    evaluate.py                   # metrics, confusion matrix, seasonal error breakdown
    plots.py                      # trend / anomaly / model figures
    pipeline.py                   # end-to-end orchestration
  scripts/run_pipeline.py         # CLI entry point
  app/dashboard.py                # Streamlit dashboard (UI)
  notebooks/weather_extremes_walkthrough.ipynb
  tests/test_pipeline.py
  data/{raw,interim,processed}/   # downloaded + derived data (git-ignored)
  models/                         # fitted pipelines (joblib)
  reports/{figures,metrics}/      # generated plots and metrics
```

---

## Methodology

1. **Data foundation.** Download `ghcnd-stations.txt` and the station `.dly` file,
   parse the fixed-width GHCN format, drop quality-flagged values (`QFLAG` set),
   convert tenths to metric units, and align to a continuous daily index.
2. **Feature engineering.**
   - *Clean:* missing/flagged handling, unit standardisation, date alignment.
   - *Calendar:* day-of-year, month, season, cyclical sine/cosine signals.
   - *Rolling:* 3-day and 7-day means, volatility, and precipitation context.
   - *Lagged:* temperature lags (1, 2, 3, 7 days) and lagged anomalies.
   - *Anomaly:* seasonal (day-of-year) baseline comparison with a project threshold.
   - All features are strictly causal (rolling stats are shifted by one day) so future
     information never leaks into the past.
3. **Labeling.** The TMAX anomaly versus the smoothed day-of-year climatology is
   thresholded at the configured quantiles (default 95th / 5th) into `extreme_hot`,
   `extreme_cold`, and `normal`.
4. **Models.** Logistic regression (standardised) and random forest, trained with class
   balancing on a chronological 60/20/20 train/validation/test split.
5. **Evaluation.** Accuracy, precision, recall, F1, ROC-AUC (one-vs-rest macro and
   extreme-vs-normal), confusion matrix, alert-threshold performance, seasonal error
   breakdown, feature importance / coefficients, and trend diagnostics.

---

## Outputs

- `reports/figures/` - temperature trend, seasonal climatology, anomaly distribution,
  annual trend, event frequency, confusion matrix, ROC curves, feature importance,
  seasonal error breakdown, probability timeline.
- `reports/metrics/` - per-model JSON metrics, test predictions, feature importances,
  coefficient table, seasonal breakdown, and `run_summary.json`.
- `models/` - serialised fitted pipelines.

---

## Example results

Default configuration (NY City Central Park, 2000-2023, 95th/5th quantiles):

| Metric | Value |
| --- | --- |
| Accuracy | 0.905 |
| F1 (macro) | 0.460 |
| F1 (extreme, macro) | 0.216 |
| ROC-AUC (extreme vs normal) | 0.824 |
| ROC-AUC (OVR macro) | 0.874 |
| TMAX trend | +0.16 deg C / decade |
| TMIN trend | +0.26 deg C / decade |

Because extreme days are rare (about 5% per class), accuracy is dominated by the
`normal` class; recall, per-class F1 and ROC-AUC are the informative metrics. The
detector predicts today's extreme status from **past-only** signals, which is a hard,
leakage-free forecasting task.

---

## Configuration reference (`config.yaml`)

| Key | Meaning |
| --- | --- |
| `data.station_id` | GHCN station id to analyse |
| `data.start_date` / `end_date` | analysis window |
| `data.elements` | variables to ingest |
| `data.download` | download from NOAA (else use cached files) |
| `features.rolling_windows` / `lag_days` | rolling and lag windows |
| `features.climatology_window` | +/- days used to smooth the seasonal baseline |
| `features.include_current_observation` | include the current-day anomaly (diagnostic, non-causal) |
| `labeling.hot_quantile` / `cold_quantile` | extreme thresholds |
| `split.*_frac` | chronological split fractions |
| `model.models` / `model.primary` | models to train and the reported model (`auto` selects by validation extreme-F1) |
| `evaluation.alert_threshold` | operational alert probability threshold |

---

## Troubleshooting

- **"Can't reach this page"** - the Streamlit server is not running. Make sure the
  terminal that launched it is still open and shows `Local URL: http://localhost:8501`.
  On a very first run, Streamlit can ask for an email address; the bundled
  `.streamlit/config.toml` disables that prompt, and passing `--server.headless true`
  also avoids it.
- **First load appears to hang** - the app compiles on first start and can take up to
  ~30 seconds. Wait, then refresh.
- **`streamlit` / `jupyter` not recognized** - activate the virtual environment, or use
  `python -m streamlit run app/dashboard.py`.
- **Port already in use** - Streamlit automatically selects the next free port; read the
  `Local URL` printed in the terminal and open that.
- **Dashboard shows "No metrics found"** - run `python scripts/run_pipeline.py` first
  (or click **Run / refresh pipeline** in the sidebar).
- **Activation blocked by execution policy** - run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

---

## Next steps

- Extend from one station to multi-station and regional comparisons.
- Test calibrated thresholds, class balancing and stronger time-series models.
- Add explainability, uncertainty bands and drift monitoring.
- Package the workflow as a reproducible dashboard or alerting tool.
