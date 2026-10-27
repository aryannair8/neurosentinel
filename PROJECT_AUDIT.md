# Project Audit (record of what was actually run)

Run date: 2026-10-26 · Windows 11 · Python 3.13 venv · TensorFlow 2.21.0 / Keras 3.15.1 · yfinance 1.7.0

## Execution
| Item | Status |
|---|---|
| Notebook executed top to bottom via `nbconvert --execute` | **Yes, no errors** |
| Live yfinance retrieval (notebook §5, §22) | **Worked**: 3,373 raw rows, 2013-02-01 → 2026-10-26 |
| Saved model reloaded and reproduces in-memory predictions | **Yes** (atol 1e-5) |
| Dashboard produced a forecast from live data | **Yes**, verified headlessly with `streamlit.testing.AppTest` (0 exceptions, 8 metrics rendered). Not yet viewed by a human in a browser. |
| Dashboard live-failure path | Implemented (falls back to saved snapshot, labelled); **not exercised** with the network disabled |
| Python 3.14 | Not used; TensorFlow could not be installed there, 3.13 venv used |

## Tests: 30 passed, 0 failed
Source of truth: `artifacts/test_results.csv` (written by the notebook). Groups: data (6), features incl. future-perturbation leakage test (5), target & split (8), model (4), dashboard code parity (1), artifacts (6).

## Verified results (from `artifacts/`)
Test period 2024-10-15 → 2026-10-23, n = 501.

| Model | MAE | RMSE | DA |
|---|---:|---:|---:|
| Zero-return | 0.006197 | 0.008310 | n/a |
| Linear regression | 0.006312 | 0.008444 | 49.1% |
| GRU (adam + huber, seed 1) | 0.006423 | 0.008649 | 49.1% |

- The GRU had the best **validation** RMSE (0.007311 vs 0.007535 zero) but **worse test** RMSE than both baselines. Validation advantage did not carry over (consistent with weak signal / selection on one window).
- Exploratory DM-style test (unadjusted): zero-return has significantly lower squared error than the GRU (p ≈ 0.003); GRU vs linear p ≈ 0.07.
- High-vol regime (74 sessions): RMSE zero 0.011498, linear 0.011807, GRU 0.012391.
- Conclusion supportable from this run: **no forecasting advantage for the GRU was demonstrated.** No profitability or backtest claim is made.
- The earlier "preliminary" notes (74 high-vol sessions, zero-baseline high-vol RMSE ≈ 0.0115) are consistent with this run; other preliminary numbers differ because the data window now extends to Oct 2026 and the selected config differs (seed 1 adam-huber vs adam-mse). Use the table above.

## Known limitations / open items
- Index volume from Yahoo contains zeros (29 rows); treated as missing, `vol_ratio` falls back to 1.0 there.
- Training is not bit-for-bit reproducible across machines; re-running re-downloads live data, so numbers will shift slightly.
- Slides and (if required) a written report are **not** produced yet.
