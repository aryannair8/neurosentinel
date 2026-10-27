# NeuroSentinel

ANN-based framework for next-day **NIFTY 50 log-return** forecasting: a compact GRU vs. a zero-return baseline and linear regression, evaluated on a strictly chronological out-of-sample test set. Individual ESE project, CHRIST (Deemed to be University), Delhi NCR.

> Experimental research prototype. Not financial advice, not a trading system.

## Structure
```
NeuroSentinel_Complete.ipynb   all code + explanations + tests + results (the main document)
app.py                         Streamlit dashboard (loads saved model; never retrains)
artifacts/                     everything produced by the notebook run (model, scaler, metadata, CSVs, test_results.csv)
tools/build_notebook.py        regenerates the notebook source (optional)
requirements.txt               pinned, tested on Python 3.13
PROJECT_AUDIT.md              what was run and what passed
.streamlit/config.toml         dark theme
```

## Setup (Python 3.13)
```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run
Notebook (re-trains all 12 configurations, ~5 min on CPU; needs internet for yfinance):
```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute NeuroSentinel_Complete.ipynb --output NeuroSentinel_Complete.ipynb --ExecutePreprocessor.timeout=3000
```
or open it in Jupyter/VS Code and *Run All*.

Dashboard:
```bash
.venv\Scripts\python -m streamlit run app.py
```

## Method in brief
- Data: `^NSEI` daily bars from Yahoo Finance (live; a raw snapshot is saved in `artifacts/`). Zero-reported volume is treated as missing; an unfinished same-day bar is dropped.
- Target: `y_t = ln(C_{t+1}/C_t)`. 14 past-only features (lagged returns, rolling vol, momentum, HL range, OC return, volume ratio), 20-session window.
- Split 70/15/15 chronological; scaler fit on training rows only; training target scaled by train std (metrics reported in original units).
- GRU(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(1), L2 1e-4; grid of Adam/RMSprop × MSE/Huber × seeds 0,1,2; selection by **validation RMSE**; test set evaluated once.
- 30 executable checks, including a future-perturbation leakage test. Results in `artifacts/test_results.csv`.

## Results
See §18–19 of the notebook and `artifacts/comparison_test.csv`. Read them as the run produced them: in the recorded run the GRU did **not** beat the zero-return baseline on test RMSE. That is a legitimate finding, not a bug.

## Limitations
Weak signal in daily returns; non-stationarity; OHLC-only features; unreliable index volume from the provider; selection bias from 12 configs; single test split; accuracy ≠ profitability; the live dashboard feeds new data to a *fixed* model and does not retrain it.

## References
Cho et al. 2014 (GRU); Diebold & Mariano 1995; Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*; yfinance, TensorFlow/Keras, scikit-learn docs.
