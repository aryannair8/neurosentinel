"""NeuroSentinel dashboard: saved-model inference on live NIFTY 50 data.

Run:  streamlit run app.py
The model is loaded from artifacts/ and is NEVER retrained here.
build_features / clean_ohlcv are copies of the notebook functions (parity is tested in the notebook).
"""
import json
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

ART = Path(__file__).parent / "artifacts"
IST = timezone(timedelta(hours=5, minutes=30))
CYAN, BLUE, GREY, RED, GREEN = "#22d3ee", "#3b82f6", "#94a3b8", "#f87171", "#34d399"


# ---------------------------------------------------------------- shared pipeline (copy of notebook)
def clean_ohlcv(raw, drop_incomplete_today=True):
    rep = {"raw_rows": len(raw)}
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    rep["duplicate_dates"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()
    px = ["Open", "High", "Low", "Close"]
    bad = df[px].isna().any(axis=1) | (df[px] <= 0).any(axis=1)
    rep["invalid_ohlc_rows_dropped"] = int(bad.sum())
    df = df[~bad]
    incons = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    rep["ohlc_inconsistent_rows_kept"] = int(incons.sum())
    rep["zero_or_missing_volume_rows"] = int((~(df["Volume"] > 0)).sum())
    df["Volume"] = df["Volume"].where(df["Volume"] > 0)
    now = datetime.now(IST)
    rep["incomplete_session_dropped"] = 0
    if drop_incomplete_today and len(df) and df.index[-1].date() == now.date() and now.time() < dtime(16, 0):
        df = df.iloc[:-1]
        rep["incomplete_session_dropped"] = 1
    rep["clean_rows"] = len(df)
    return df, rep


def build_features(df, add_target=True):
    d = pd.DataFrame(index=df.index)
    c = df["Close"]
    r = np.log(c / c.shift(1))
    for k in range(5):
        d[f"ret_lag_{k}"] = r.shift(k)
    for w in (5, 10, 20):
        d[f"vol_{w}"] = r.rolling(w).std()
    for w in (5, 10, 20):
        d[f"mom_{w}"] = np.log(c / c.shift(w))
    d["hl_range"] = (df["High"] - df["Low"]) / c
    d["oc_return"] = np.log(c / df["Open"])
    v = df["Volume"].where(df["Volume"] > 0)
    ref = v.shift(1).rolling(20, min_periods=10).mean()
    d["vol_ratio"] = (v / ref).fillna(1.0)
    if add_target:
        d["target"] = r.shift(-1)
    return d
# ---------------------------------------------------------------- end shared pipeline


from PIL import Image

LIME, LIME2, LIME3, PALE = "#b4f94f", "#6cc400", "#3f7a00", "#eefed4"
CARD, BG, MUTED, RED = "#2a2a2a", "#111111", "#9a9a9a", "#ff7a7a"
LOGO = Image.open(Path(__file__).parent / "assets" / "logo.png")

st.set_page_config(page_title="NeuroSentinel", page_icon=LOGO, layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input {{font-family:'Poppins',sans-serif !important;}}
.stApp {{background:{BG};}}
header[data-testid="stHeader"] {{background:transparent;}}
.block-container {{padding-top:2.2rem; max-width:1280px;}}
section[data-testid="stSidebar"] {{background:#0a0a0a; border-right:1px solid #333;}}
section[data-testid="stSidebar"] > div {{padding-top:1.2rem;}}
/* pill navigation */
[data-testid="stSidebar"] [role="radiogroup"] {{gap:6px;}}
[data-testid="stSidebar"] [role="radiogroup"] label {{padding:10px 20px; border-radius:999px; width:100%; cursor:pointer;}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{display:none;}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{color:{MUTED}; font-size:1rem;}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {{color:#fff;}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{background:{LIME};}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{color:#111; font-weight:600;}}
/* cards */
.card {{background:{CARD}; border-radius:24px; padding:24px 28px; height:100%; min-height:150px;}}
.card .lbl {{color:#ddd; font-size:1rem; font-weight:400;}}
.card .val {{color:#fff; font-size:1.8rem; font-weight:600; line-height:1.25; margin:4px 0; white-space:nowrap;}}
.card .val.acc {{color:{LIME};}}
.card .sub {{color:#bbb; font-size:.85rem; font-weight:300;}}
[data-testid="stVerticalBlockBorderWrapper"] {{background:{CARD}; border:none !important; border-radius:24px; padding:10px 14px;}}
.h1 {{color:#fff; font-size:2.5rem; font-weight:700; letter-spacing:-.02em; line-height:1.1; margin:0;}}
.h1sub {{color:{MUTED}; font-weight:300; margin:6px 0 22px 0;}}
.h2 {{color:#fff; font-size:1.5rem; font-weight:600; margin:6px 0 2px 4px;}}
.pill {{display:inline-block; background:{LIME}; color:#111; font-weight:500; padding:8px 20px; border-radius:999px; font-size:.95rem;}}
.pill.warn {{background:#fbbf24;}} .pill.ghost {{background:#2a2a2a; color:#ddd;}}
.brand {{display:flex; align-items:center; gap:12px; margin-bottom:6px;}}
.brand b {{color:#fff; font-size:1.25rem; font-weight:600;}} .brand span {{color:{MUTED}; font-size:.8rem; font-weight:300; display:block;}}
.legend {{display:flex; align-items:center; gap:10px; color:#ddd; font-weight:300; margin:6px 0;}}
.dot {{width:9px; height:9px; border-radius:50%; display:inline-block;}}
hr.thin {{border:none; border-top:1px solid #777; margin:14px 0;}}
[data-testid="stDataFrame"] {{border-radius:16px; overflow:hidden;}}
div[data-testid="stAlert"] {{border-radius:16px;}}
</style>
""",
    unsafe_allow_html=True,
)


def card(label, value, sub="", accent=False):
    st.markdown(f'<div class="card"><div class="lbl">{label}</div><div class="val {"acc" if accent else ""}">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def header(title, sub, pill_html=""):
    a, b = st.columns([3, 1])
    a.markdown(f'<div class="h1">{title}</div><div class="h1sub">{sub}</div>', unsafe_allow_html=True)
    b.markdown(f'<div style="text-align:right; padding-top:14px">{pill_html}</div>', unsafe_allow_html=True)


def style(fig, h=340):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6, r=6, t=40, b=6), legend=dict(orientation="h", y=-0.12, x=0), font=dict(family="Poppins"),
                      title_font=dict(size=15, color="#fff"))
    fig.update_xaxes(gridcolor="#3a3a3a", zeroline=False)
    fig.update_yaxes(gridcolor="#3a3a3a", zeroline=False)
    return fig


def show(fig, h=340):
    with st.container():
        st.plotly_chart(style(fig, h), width="stretch")


@st.cache_resource(show_spinner="Loading saved model...")
def load_artifacts():
    import keras
    meta = json.loads((ART / "metadata.json").read_text())
    model = keras.saving.load_model(ART / "gru_model.keras", compile=False)
    scaler = joblib.load(ART / "scaler.joblib")
    return meta, model, scaler


@st.cache_data(show_spinner=False)
def load_tables():
    names = ["comparison_val", "comparison_test", "volatility_regime_test", "experiments"]
    out = {n: pd.read_csv(ART / f"{n}.csv") for n in names if (ART / f"{n}.csv").exists()}
    tp = ART / "test_predictions.csv"
    if tp.exists():
        out["test_predictions"] = pd.read_csv(tp, parse_dates=["date"])
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live(ticker):
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    try:
        start = (datetime.now() - timedelta(days=220)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("Provider returned no rows.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw, stamp, None
    except Exception as exc:  # network / provider failure: surfaced to the UI, never hidden
        return None, stamp, f"{type(exc).__name__}: {exc}"


def forecast_latest(df_clean, model, scaler, meta):
    feats = build_features(df_clean, add_target=False).dropna(subset=meta["features"])
    L = meta["lookback"]
    if len(feats) < L:
        raise ValueError(f"Need {L} complete feature rows, have {len(feats)}.")
    win = feats[meta["features"]].iloc[-L:]
    X = scaler.transform(win.values)[None, :, :]
    yhat = float(model.predict(X, verbose=0).ravel()[0] * meta["y_scale"])
    asof = win.index[-1]
    return yhat, asof, float(df_clean.loc[asof, "Close"])


# ------------------------------------------------------------------ load
if not (ART / "metadata.json").exists():
    st.error("artifacts/ is missing. Run NeuroSentinel_Complete.ipynb top to bottom first.")
    st.stop()
meta, model, scaler = load_artifacts()
T = load_tables()

PAGES = ["Overview", "Forecast Lab", "Model Arena", "Diagnostics", "Data & Method"]
with st.sidebar:
    c1, c2 = st.columns([1, 2.4])
    c1.image(LOGO, width=56)
    c2.markdown('<div class="brand" style="margin-top:6px"><div><b>NeuroSentinel</b><span>NIFTY 50 · GRU research</span></div></div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    include_open = st.checkbox("Include in-progress session", value=False,
                               help="Off: today's unfinished daily bar is dropped before forecasting.")
    if st.button("Fetch latest data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.caption(f"Model trained {meta['trained_at_utc']}")
    st.caption(f"Training data through {meta['splits']['train']['end']}")
    st.caption("Experimental prototype · not financial advice")

raw, fetched_at, err = fetch_live(meta["ticker"])
live_ok = raw is not None
if live_ok:
    df, rep = clean_ohlcv(raw, drop_incomplete_today=not include_open)
    source_label = "LIVE · Yahoo Finance"
else:
    snap = ART / "raw_data_snapshot.csv"
    df, rep = (clean_ohlcv(pd.read_csv(snap, index_col=0, parse_dates=True), drop_incomplete_today=False)
               if snap.exists() else (None, {}))
    source_label = "SAVED SNAPSHOT · not live"
status_pill = f'<span class="pill {"" if live_ok else "warn"}">{source_label}</span>'

fc, fc_err = None, ""
if df is not None and len(df):
    try:
        fc = forecast_latest(df, model, scaler, meta)
    except Exception as exc:
        fc_err = str(exc)

ct = T.get("comparison_test")
fmt = lambda d: d.round(6).astype(object).where(d.notna(), "n/a")  # undefined DA (zero baseline) shown as n/a
rmse = ct.set_index("model")["RMSE"] if ct is not None else None

# ------------------------------------------------------------------ Overview
if page == "Overview":
    header("Dashboard Overview", f"Here's your NIFTY 50 forecast overview · data retrieved {fetched_at}", status_pill)
    if err:
        st.error(f"Live retrieval failed: {err}. Showing the saved snapshot only; no live forecast is available.")
    a, b, c = st.columns(3)
    with a:
        if df is not None:
            card("Latest Close", f"{df['Close'].iloc[-1]:,.2f}", f"Session {df.index[-1].date()}", accent=True)
    with b:
        if fc:
            card("Next-Session Forecast", f"{fc[0] * 100:+.4f}%", "GRU log-return estimate (model output)")
    with c:
        if rmse is not None:
            d = (rmse["GRU"] / rmse["Zero-return"] - 1) * 100
            card("GRU Test RMSE", f"{rmse['GRU']:.5f}", f"{d:+.1f}% vs zero-return baseline ({rmse['Zero-return']:.5f})")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        st.markdown('<div class="h2">Test Performance</div>', unsafe_allow_html=True)
        if ct is not None and "test_predictions" in T:
            tp = T["test_predictions"]
            n_hv = int((tp.regime == "high_vol").sum())
            col1, col2 = st.columns([1, 1])
            fig = go.Figure(go.Pie(labels=["Normal", "High-vol"], values=[len(tp) - n_hv, n_hv], hole=0.62, sort=False,
                                   marker=dict(colors=[PALE, LIME2], line=dict(color=CARD, width=3)), textinfo="none"))
            fig.update_layout(showlegend=False)
            with col1:
                show(fig, 280)
            with col2:
                st.markdown(f'<div class="lbl" style="color:#ddd">Test sessions</div><div class="val acc" style="font-size:2rem;font-weight:600;color:{LIME}">{len(tp)}</div>'
                            f'<div class="sub" style="color:#bbb;font-weight:300">{meta["splits"]["test"]["start"]} → {meta["splits"]["test"]["end"]}</div><hr class="thin">'
                            f'<div class="legend"><span class="dot" style="background:{PALE}"></span>Normal vol ({len(tp) - n_hv})</div>'
                            f'<div class="legend"><span class="dot" style="background:{LIME2}"></span>High vol ({n_hv})</div>',
                            unsafe_allow_html=True)
            st.dataframe(fmt(ct), hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.markdown('<div class="h2">NIFTY 50 · 60 sessions</div>', unsafe_allow_html=True)
        if df is not None:
            last = df["Close"].iloc[-60:]
            fig = go.Figure(go.Scatter(x=last.index, y=last.values, mode="lines", line=dict(color=LIME, width=2.5, shape="spline"),
                                       fill="tozeroy", fillcolor="rgba(180,249,79,0.06)"))
            fig.update_yaxes(range=[float(last.min()) * 0.99, float(last.max()) * 1.01])
            show(fig, 300)
        st.markdown(
            '<div class="sub" style="color:#bbb;font-weight:300;padding:0 6px 8px">Daily returns carry little signal. A GRU is not guaranteed to beat a zero-return benchmark, '
            'and accuracy is not profitability. New data go through a <b>fixed</b> model; nothing is retrained here.</div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ Forecast Lab
elif page == "Forecast Lab":
    header("Forecast Lab", "Saved GRU applied to the latest available observations", status_pill)
    if df is None or not len(df):
        st.warning("No data available.")
    elif fc is None:
        st.error(f"Could not produce a forecast: {fc_err}")
    else:
        yhat, asof, close_t = fc
        implied = close_t * float(np.exp(yhat))
        a, b, c, d = st.columns(4)
        with a: card("Input Data As Of", str(asof.date()), f"Retrieved {fetched_at}")
        with b: card("Last Close (C_t)", f"{close_t:,.2f}", "Latest completed session")
        with c: card("Predicted Log Return", f"{yhat * 100:+.4f}%", "Next trading session", accent=True)
        with d: card("Implied Next Close", f"{implied:,.2f}", f"{implied - close_t:+,.2f} · estimate, not a guarantee")
        if not live_ok:
            st.warning("This forecast uses the SAVED SNAPSHOT because live retrieval failed. It is not a live forecast.")
        st.caption(f"Forecast generated {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}. Model trained on data through "
                   f"{meta['splits']['train']['end']}; it has not been retrained on newer observations.")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            last = df.iloc[-90:]
            fig = go.Figure(go.Candlestick(x=last.index, open=last.Open, high=last.High, low=last.Low, close=last.Close,
                                           increasing_line_color=LIME, decreasing_line_color=RED, name="NIFTY 50"))
            nxt = asof + pd.tseries.offsets.BDay(1)
            fig.add_trace(go.Scatter(x=[asof, nxt], y=[close_t, implied], mode="lines+markers", line=dict(color="#fff", dash="dot"),
                                     name="Model-implied next close"))
            fig.update_layout(xaxis_rangeslider_visible=False, title="Last 90 sessions + implied next close (estimate)")
            show(fig, 430)
        l, r = st.columns([1, 1])
        with l, st.container(border=True):
            st.markdown('<div class="h2">Latest observations</div>', unsafe_allow_html=True)
            st.dataframe(df.tail(10).round(2).iloc[::-1], width="stretch")
        with r, st.container(border=True):
            st.markdown('<div class="h2">Stored forecasts · historical, not live</div>', unsafe_allow_html=True)
            if "test_predictions" in T:
                tp = T["test_predictions"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tp.date, y=tp.actual, name="Actual", line=dict(color="#777", width=1)))
                fig.add_trace(go.Scatter(x=tp.date, y=tp.gru, name="GRU", line=dict(color=LIME, width=1.8)))
                show(fig, 300)

# ------------------------------------------------------------------ Model Arena
elif page == "Model Arena":
    header("Model Arena", "Zero-return vs linear regression vs GRU · same test targets, dates and metrics", "")
    if ct is not None:
        cols = st.columns(2)
        for col, m in zip(cols, ("RMSE", "MAE")):
            with col, st.container(border=True):
                fig = go.Figure(go.Bar(x=ct["model"], y=ct[m], marker_color=["#6b6b6b", "#9a9a9a", LIME][: len(ct)],
                                       text=ct[m].round(6), textposition="outside"))
                fig.update_layout(title=f"Test {m} · lower is better")
                fig.update_yaxes(range=[float(ct[m].min()) * 0.9, float(ct[m].max()) * 1.06])
                show(fig, 320)
        with st.container(border=True):
            st.markdown('<div class="h2">Test set</div>', unsafe_allow_html=True)
            st.dataframe(fmt(ct), hide_index=True, width="stretch")
        if "comparison_val" in T:
            with st.container(border=True):
                st.markdown('<div class="h2">Validation set</div>', unsafe_allow_html=True)
                st.dataframe(fmt(T["comparison_val"]), hide_index=True, width="stretch")
        if "experiments" in T:
            with st.container(border=True):
                st.markdown('<div class="h2">Experiment grid · selection by validation RMSE</div>', unsafe_allow_html=True)
                st.dataframe(T["experiments"].sort_values("val_rmse"), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Diagnostics
elif page == "Diagnostics":
    header("Diagnostics", "Residual analysis and volatility regimes on the held-out test period", "")
    if "test_predictions" in T:
        tp = T["test_predictions"].copy()
        tp["resid"] = tp.actual - tp.gru
        specs = [
            ("Actual vs predicted", go.Scatter(x=tp.gru, y=tp.actual, mode="markers", marker=dict(color=LIME, size=4, opacity=0.6))),
            ("Residual distribution", go.Histogram(x=tp.resid, nbinsx=50, marker_color=LIME2)),
            ("Residuals over time", go.Scatter(x=tp.date, y=tp.resid, mode="lines", line=dict(color=LIME, width=1))),
            ("Residuals vs predictions", go.Scatter(x=tp.gru, y=tp.resid, mode="markers", marker=dict(color=PALE, size=4, opacity=0.6))),
        ]
        for i in range(0, 4, 2):
            cols = st.columns(2)
            for col, (title, trace) in zip(cols, specs[i:i + 2]):
                with col, st.container(border=True):
                    fig = go.Figure(trace)
                    fig.update_layout(title=title)
                    show(fig, 300)
        with st.container(border=True):
            fig = go.Figure(go.Histogram(x=tp.actual, nbinsx=60, marker_color="#8a8a8a"))
            fig.update_layout(title="Test-period return distribution")
            show(fig, 260)
    if "volatility_regime_test" in T:
        with st.container(border=True):
            st.markdown('<div class="h2">Volatility regimes</div>', unsafe_allow_html=True)
            st.caption(f"High-vol = 20-day volatility above the {meta['config']['high_vol_quantile']:.0%} quantile of the TRAINING period "
                       f"(threshold {meta['high_vol_threshold']:.5f}); fixed in advance.")
            st.dataframe(fmt(T["volatility_regime_test"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Data & Method
else:
    header("Data & Method", "What the model sees, how it was trained, and what a forecast means", "")
    with st.container(border=True):
        st.markdown(f"""
**Data.** Daily `{meta['ticker']}` (NIFTY 50) via Yahoo Finance / `yfinance`, {meta['data_start']} → {meta['data_end']}
({meta['n_clean_rows']} cleaned sessions at training time). Daily bars, not guaranteed real-time.

**Target.** `y_t = ln(C_(t+1) / C_t)`, the next session's log return, from information available at the close of session *t*.

**Features ({len(meta['features'])}).** `{', '.join(meta['features'])}`

**Window.** Each prediction uses the previous **{meta['lookback']}** sessions of these features, standardised with a scaler fitted on the training period only.

**Model.** GRU({meta['config']['gru_units']}) → Dropout({meta['config']['dropout']}) → Dense({meta['config']['dense_units']}, ReLU) → Dense(1), L2 = {meta['config']['l2']}.
Targets are divided by the training-set std ({meta['y_scale']:.6f}) during training and multiplied back at inference.

**Split.** Chronological {meta['config']['train_frac']:.0%} / {meta['config']['val_frac']:.0%} / {1 - meta['config']['train_frac'] - meta['config']['val_frac']:.0%};
validation RMSE selects among {meta['n_experiments']} configurations; the test set is touched once.

**Selected.** {meta['selected']['optimizer']} + {meta['selected']['loss']}, seed {meta['selected']['seed']} (validation RMSE {meta['selected']['val_rmse']:.6f}).

**Meaning of a forecast.** A noisy estimate of tomorrow's log return from a fixed model. It is not a trading signal, not a guarantee, and says nothing about profitability after costs.
""")
