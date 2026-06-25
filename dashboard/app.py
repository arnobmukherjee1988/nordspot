"""SE3 Day-Ahead Price Forecast Dashboard — v2.

Run with:
    streamlit run dashboard/app.py
"""

import json
import sys
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db.schema import SERIES, init_schema
from ml.models import lear, lgbm
from pipeline.features import build_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SE3 Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2563eb"
BAND   = "rgba(99,153,255,0.15)"
ACTUAL = "#111111"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
GRAY   = "#888888"

st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { min-width: 230px; max-width: 230px; }

    /* Metric cards */
    .metric-card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px; padding: 16px 20px; text-align: center;
    }
    .metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 600; }
    .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }

    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid rgba(128,128,128,0.15);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 18px;
        border-radius: 8px 8px 0 0;
        font-size: 14px;
        font-weight: 500;
        color: #888;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2.5px solid #2563eb !important;
        background: rgba(37,99,235,0.04) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563eb;
        background: rgba(37,99,235,0.06) !important;
    }

    /* Bitemporal badge */
    .bt-badge {
        display: inline-block;
        background: rgba(37,99,235,0.1);
        color: #2563eb;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", color=BLUE):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def apply_layout(fig, title, y_label="EUR/MWh", height=420):
    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title=y_label,
        yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        template="plotly_white",
        height=height,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def _band_trace(x, lo, hi, name):
    return go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor=BAND,
        line=dict(color="rgba(0,0,0,0)"),
        name=name, hoverinfo="skip",
    )


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading data from TimeDB …")
def load_features_full():
    """Load 180 + 14-day window once. Slider slices in memory."""
    td    = init_schema()
    end   = pd.Timestamp.utcnow().floor("h")
    start = end - pd.Timedelta(days=180 + 14)
    return build_features(td, start.to_pydatetime(), end.to_pydatetime())


@st.cache_data(ttl=3600, show_spinner="Loading backtest data …")
def load_features_range(from_str: str, to_str: str) -> pd.DataFrame:
    """Load with 14-day warm-up so all lag features are populated."""
    td     = init_schema()
    start  = datetime.fromisoformat(from_str).replace(tzinfo=timezone.utc)
    end    = datetime.fromisoformat(to_str).replace(tzinfo=timezone.utc) + timedelta(days=1)
    warmup = start - timedelta(days=14)
    df_full = build_features(td, warmup, end)
    return df_full[df_full.index >= pd.Timestamp(start)]


@st.cache_data(ttl=1800, show_spinner=False)
def load_stored_forecasts_full():
    """Read all stored forecast quantiles from TimeDB (full history, no time filter)."""
    td = init_schema()
    all_ids = [
        SERIES["lgbm_q05"], SERIES["lgbm_q50"], SERIES["lgbm_q95"],
        SERIES["lear_q05"],  SERIES["lear_q50"],  SERIES["lear_q95"],
    ]
    try:
        raw = td.read(series_ids=all_ids, retention="forever")
        pdf = raw.to_pandas()
        if len(pdf) == 0:
            return None, None
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
        pivot = pdf.pivot_table(
            index="valid_time", columns="series_id", values="value", aggfunc="last"
        )
        pivot.index.name = "valid_time"
        id_map = {v: k for k, v in SERIES.items()}
        pivot.columns = [id_map.get(int(c), str(c)) for c in pivot.columns]
        lgbm_fc = pivot.reindex(columns=["lgbm_q05", "lgbm_q50", "lgbm_q95"])
        lear_fc  = pivot.reindex(columns=["lear_q05",  "lear_q50",  "lear_q95"])
        if lgbm_fc["lgbm_q50"].notna().sum() == 0:
            return None, None
        return lgbm_fc, lear_fc if lear_fc["lear_q50"].notna().sum() > 0 else None
    except Exception:
        return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def load_stored_forecasts_with_kt():
    """Read stored p50 forecasts with knowledge_time included.

    Returns a long-form DataFrame with columns:
        series_id, knowledge_time, valid_time, value
    Used to show *when* each forecast was produced (bitemporal audit).
    """
    td = init_schema()
    try:
        raw = td.read(
            series_ids=[SERIES["lgbm_q50"], SERIES["lear_q50"]],
            retention="forever",
            include_knowledge_time=True,
        )
        pdf = raw.to_pandas()
        if len(pdf) == 0:
            return None
        pdf["valid_time"]     = pd.to_datetime(pdf["valid_time"],     utc=True)
        pdf["knowledge_time"] = pd.to_datetime(pdf["knowledge_time"], utc=True)
        id_map = {v: k for k, v in SERIES.items()}
        pdf["model"] = pdf["series_id"].map(lambda x: id_map.get(int(x), str(x)))
        return pdf
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_forecasts_as_of(as_of_iso: str, from_iso: str, to_iso: str):
    """As-of query: what forecasts existed with knowledge_time < as_of?

    This is the core bitemporal read. It answers: "what did the model predict
    for this period, if you could only use information available at as_of_iso?"
    """
    td    = init_schema()
    as_of = datetime.fromisoformat(as_of_iso).replace(tzinfo=timezone.utc)
    start = datetime.fromisoformat(from_iso).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(to_iso).replace(tzinfo=timezone.utc) + timedelta(days=1)
    all_ids = [
        SERIES["lgbm_q05"], SERIES["lgbm_q50"], SERIES["lgbm_q95"],
        SERIES["lear_q05"],  SERIES["lear_q50"],  SERIES["lear_q95"],
    ]
    try:
        raw = td.read(
            series_ids=all_ids,
            retention="forever",
            end_known=as_of,
            start_valid=start,
            end_valid=end,
        )
        pdf = raw.to_pandas()
        if len(pdf) == 0:
            return None, None
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
        pivot = pdf.pivot_table(
            index="valid_time", columns="series_id", values="value", aggfunc="last"
        )
        pivot.index.name = "valid_time"
        id_map = {v: k for k, v in SERIES.items()}
        pivot.columns = [id_map.get(int(c), str(c)) for c in pivot.columns]
        lgbm_fc = pivot.reindex(columns=["lgbm_q05", "lgbm_q50", "lgbm_q95"])
        lear_fc  = pivot.reindex(columns=["lear_q05",  "lear_q50",  "lear_q95"])
        if lgbm_fc["lgbm_q50"].notna().sum() == 0:
            return None, None
        return lgbm_fc, lear_fc if lear_fc["lear_q50"].notna().sum() > 0 else None
    except Exception:
        return None, None


@st.cache_data(ttl=600, show_spinner=False)
def load_day_ahead_gate_closure(from_iso: str, to_iso: str):
    """True day-ahead backtest using read_relative().

    For each valid_time, returns only the forecast that existed at gate closure
    (noon D-1, SE3 convention). This is impossible to fake with a flat file —
    TimeDB enforces the knowledge_time constraint in the SQL query itself.
    """
    td    = init_schema()
    start = datetime.fromisoformat(from_iso).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(to_iso).replace(tzinfo=timezone.utc) + timedelta(days=1)
    try:
        raw = td.read_relative(
            series_ids=[SERIES["lgbm_q50"], SERIES["lear_q50"]],
            retention="forever",
            days_ahead=1,
            time_of_day=dt_time(12, 0),   # SE3 gate closure: noon D-1
            start_valid=start,
            end_valid=end,
        )
        pdf = raw.to_pandas()
        if len(pdf) == 0:
            return None, None
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"], utc=True)
        pivot = pdf.pivot_table(
            index="valid_time", columns="series_id", values="value", aggfunc="last"
        )
        pivot.index.name = "valid_time"
        id_map = {v: k for k, v in SERIES.items()}
        pivot.columns = [id_map.get(int(c), str(c)) for c in pivot.columns]
        lgbm_p50 = pivot.get("lgbm_q50")
        lear_p50  = pivot.get("lear_q50")
        return lgbm_p50, lear_p50
    except Exception:
        return None, None


def load_metrics() -> dict | None:
    path = Path("model/metrics.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def models_available() -> bool:
    return (Path("model/lgbm_q50.pkl").exists() and
            Path("model/lear_h00.pkl").exists())


def get_td():
    return init_schema()


# ── Sidebar — controls only ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ SE3 Forecast")
    st.markdown("---")
    days_back = st.slider("History window (days)", 7, 180, 60, 1)
    show_lear = st.checkbox("Show LEAR benchmark", value=True)
    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    m_all = load_metrics()
    if m_all:
        m = m_all.get("lgbm", {})
        st.caption(f"LightGBM MAE: **{m.get('mae', 0):.2f}** EUR/MWh")
    st.markdown("---")
    st.caption("Data: ENTSO-E · Open-Meteo\nModel: LightGBM · LEAR\n\nBuilt by Arnob Mukherjee")


# ── Shared data — load once, slice in memory ───────────────────────────────────

with st.spinner("Loading data …"):
    _df_full                             = load_features_full()
    _stored_lgbm_full, _stored_lear_full = load_stored_forecasts_full()

_cutoff     = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(days=days_back)
df          = _df_full[_df_full.index >= _cutoff]
stored_lgbm = _stored_lgbm_full[_stored_lgbm_full.index >= _cutoff] if _stored_lgbm_full is not None else None
stored_lear  = _stored_lear_full[_stored_lear_full.index  >= _cutoff] if _stored_lear_full  is not None else None

has_prices = df["price"].notna().sum() > 24
has_models = models_available()
has_stored = stored_lgbm is not None

lgbm_pred = lear_pred = None
if has_models and has_prices:
    lgbm_pred = lgbm.predict(df)
    if show_lear:
        lear_pred = lear.predict(df)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_fc, tab_perf, tab_bt, tab_diag, tab_data, tab_about = st.tabs([
    "📈 Forecast",
    "📊 Performance",
    "📉 Backtesting",
    "🔬 Diagnostics",
    "🗃️ Data Explorer",
    "ℹ️ About",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_fc:
    st.title("📈 SE3 Day-Ahead Price Forecast")

    if not has_prices:
        st.info("No electricity price data yet — ENTSO-E sync pending.")
        st.stop()

    now = pd.Timestamp.utcnow().floor("h")

    # Resolve source: TimeDB stored forecasts > live predict()
    if has_stored:
        fc_lgbm_src  = stored_lgbm
        fc_lear_src  = stored_lear
        fc_source_label = "📦 **TimeDB** — stored at training time"
    elif lgbm_pred is not None:
        fc_lgbm_src  = lgbm_pred
        fc_lear_src  = lear_pred
        fc_source_label = "⚡ Live — run `python -m ml.train` to persist"
    else:
        fc_lgbm_src = fc_lear_src = None
        fc_source_label = None

    fc_df = None
    if fc_lgbm_src is not None:
        available = fc_lgbm_src.dropna(subset=["lgbm_q50"])
        fc_df = available.tail(24) if len(available) >= 24 else (available if len(available) else None)

    if fc_source_label:
        st.caption(f"Forecast source: {fc_source_label}")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg = fc_df["lgbm_q50"].mean() if fc_df is not None else float("nan")
        metric_card("Avg forecast", f"{avg:.1f}", "EUR/MWh")
    with c2:
        if fc_df is not None:
            peak_val  = fc_df["lgbm_q50"].max()
            peak_time = fc_df["lgbm_q50"].idxmax().strftime("%H:%M")
            metric_card("Peak (p50)", f"{peak_val:.1f}", f"at {peak_time}", AMBER)
        else:
            metric_card("Peak (p50)", "—", "", AMBER)
    with c3:
        if fc_df is not None:
            night     = fc_df.loc[fc_df.index.hour.isin([0,1,2,3,4,5]), "lgbm_q50"]
            night_val = f"{night.min():.1f}" if len(night) else "—"
            metric_card("Overnight low", night_val, "EUR/MWh  00–05 h", BLUE)
        else:
            metric_card("Overnight low", "—", "", BLUE)
    with c4:
        if fc_df is not None:
            hw = ((fc_df["lgbm_q95"] - fc_df["lgbm_q05"]) / 2).mean()
            metric_card("Uncertainty", f"±{hw:.1f}", "avg half-width EUR/MWh", GRAY)
        else:
            metric_card("Uncertainty", "—", "", GRAY)

    st.markdown("---")

    # Main chart
    x_start = now - pd.Timedelta(days=days_back)
    hist    = df[df.index >= x_start]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["price"],
        name="Actual price", line=dict(color=ACTUAL, width=1.5),
    ))
    if fc_lgbm_src is not None:
        view = fc_lgbm_src[fc_lgbm_src.index >= x_start]
        fig.add_trace(_band_trace(view.index, view["lgbm_q05"], view["lgbm_q95"], "90% band"))
        fig.add_trace(go.Scatter(
            x=view.index, y=view["lgbm_q50"],
            name="LightGBM p50", line=dict(color=BLUE, width=2.5),
        ))
    if show_lear and fc_lear_src is not None:
        view_l = fc_lear_src[fc_lear_src.index >= x_start]
        fig.add_trace(go.Scatter(
            x=view_l.index, y=view_l["lear_q50"],
            name="LEAR p50", line=dict(color=AMBER, width=1.8, dash="dash"),
        ))
    apply_layout(fig, f"SE3 Price Forecast — last {days_back} days")
    st.plotly_chart(fig, use_container_width=True)

    if not has_models and not has_stored:
        st.info("💡 Run `python -m ml.train` to train models and store forecasts.")

    if fc_df is not None:
        with st.expander("📋 Hourly forecast table"):
            tbl = fc_df[["lgbm_q05", "lgbm_q50", "lgbm_q95"]].copy()
            tbl.columns = ["Lower (q5)", "Median (q50)", "Upper (q95)"]
            st.dataframe(tbl.round(2), use_container_width=True)

    # ── Bitemporal: knowledge_time audit ─────────────────────────────────────
    if has_stored:
        st.markdown("---")
        st.subheader("🕐 Forecast provenance — when was each prediction produced?")
        st.caption(
            "Each stored forecast carries a `knowledge_time` — the moment it was written "
            "to TimeDB. This is TimeDB's bitemporal dimension: valid_time is *when the price occurs*, "
            "knowledge_time is *when the model produced the forecast*."
        )

        kt_df = load_stored_forecasts_with_kt()
        if kt_df is not None:
            lgbm_kt = kt_df[kt_df["model"] == "lgbm_q50"].copy()
            if not lgbm_kt.empty:
                kt_summary = lgbm_kt.groupby("knowledge_time")["valid_time"].agg(["min", "max", "count"])
                kt_summary.index = kt_summary.index.strftime("%Y-%m-%d %H:%M UTC")
                kt_summary.columns = ["First valid_time", "Last valid_time", "Hours stored"]
                kt_summary["First valid_time"] = kt_summary["First valid_time"].dt.strftime("%Y-%m-%d %H:%M")
                kt_summary["Last valid_time"]  = kt_summary["Last valid_time"].dt.strftime("%Y-%m-%d %H:%M")
                st.markdown("**Training runs that have stored forecasts in TimeDB:**")
                st.dataframe(kt_summary, use_container_width=True)
                st.caption(
                    "Each row is a training run. A new row appears every time you run "
                    "`python -m ml.train`. TimeDB never overwrites old rows — both the new "
                    "and old forecasts coexist, queryable independently."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perf:
    st.title("📊 Model Performance")

    if not m_all:
        st.info("No metrics found. Run `python -m ml.train` first.")
        st.stop()

    model_choice = st.radio("Model", ["LightGBM", "LEAR"], horizontal=True)
    m_key = "lgbm" if model_choice == "LightGBM" else "lear"
    m = m_all.get(m_key, {})

    st.caption(f"Test set: **{m.get('test_from','?')}** → **{m.get('test_to','?')}**"
               f" ({m.get('n_hours', 0):,} hours)")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("MAE", f"{m['mae']:.2f}", "EUR/MWh")
    with c2:
        metric_card("RMSE", f"{m['rmse']:.2f}", "EUR/MWh")
    with c3:
        mape_color = GREEN if m["mape"] < 20 else AMBER if m["mape"] < 40 else RED
        metric_card("MAPE", f"{m['mape']:.1f}%", "excl. near-zero hours", mape_color)
    with c4:
        cov_color = GREEN if m["coverage_q5_q95"] >= 85 else AMBER
        metric_card("PI Coverage", f"{m['coverage_q5_q95']:.1f}%", "q5–q95  ideal ~90%", cov_color)
    with c5:
        spike_val = f"{m['spike_mae']:.1f}" if m.get("spike_mae") else "N/A"
        metric_card("Spike MAE", spike_val,
                    f"n={m.get('n_spikes',0)} hours >100 EUR/MWh", RED)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("MAE by hour of day")
        mae_hour = pd.Series(m["mae_by_hour"]).sort_index()
        mae_hour.index = mae_hour.index.astype(int)
        colors = [RED  if h in [8,9,17,18,19,20] else
                  BLUE if h in [0,1,2,3,4,5,23]  else GRAY
                  for h in mae_hour.index]
        fig = go.Figure(go.Bar(x=mae_hour.index, y=mae_hour.values, marker_color=colors))
        fig.update_layout(
            xaxis=dict(tickmode="linear"), yaxis_title="MAE (EUR/MWh)",
            template="plotly_white", height=320,
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Peak hours  🔵 Night hours  ⬛ Daytime")

    with col_r:
        st.subheader("Error breakdown")
        breakdown = pd.DataFrame({
            "Period": ["Overall", "Night (23–05 h)", "Peak (08–09, 17–20 h)"],
            "MAE (EUR/MWh)": [m["mae"], m["night_mae"], m["peak_mae"]],
        })
        fig2 = go.Figure(go.Bar(
            x=breakdown["MAE (EUR/MWh)"], y=breakdown["Period"],
            orientation="h", marker_color=[GRAY, BLUE, RED],
        ))
        fig2.update_layout(
            xaxis_title="MAE (EUR/MWh)", template="plotly_white",
            height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        if "lgbm" in m_all and "lear" in m_all:
            st.subheader("Model comparison")
            cmp = pd.DataFrame({
                "Metric":    ["MAE", "RMSE", "MAPE (%)", "Coverage (%)"],
                "LightGBM": [m_all["lgbm"]["mae"], m_all["lgbm"]["rmse"],
                             m_all["lgbm"]["mape"], m_all["lgbm"]["coverage_q5_q95"]],
                "LEAR":     [m_all["lear"]["mae"], m_all["lear"]["rmse"],
                             m_all["lear"]["mape"], m_all["lear"]["coverage_q5_q95"]],
            }).set_index("Metric")
            st.dataframe(cmp.round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_bt:
    st.title("📉 Backtesting")

    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From", value=date.today() - timedelta(days=14))
    with col2:
        to_date = st.date_input("To", value=date.today())

    if from_date >= to_date:
        st.error("From date must be before To date.")
        st.stop()
    if not has_models:
        st.info("Train models first: `python -m ml.train`")
        st.stop()

    bt_df   = load_features_range(str(from_date), str(to_date))
    bt_lgbm = lgbm.predict(bt_df)
    bt_lear  = lear.predict(bt_df) if show_lear else None

    actual  = bt_df["price"]
    p50     = bt_lgbm["lgbm_q50"]
    p05     = bt_lgbm["lgbm_q05"]
    p95     = bt_lgbm["lgbm_q95"]
    has_fc  = p50.notna() & actual.notna()
    err     = p50 - actual
    abs_err = err.abs()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Hours", f"{len(bt_df):,}", f"{from_date} → {to_date}")
    with c2:
        fc_count = int(has_fc.sum())
        metric_card("Hours with forecast", f"{fc_count:,}", f"{100*has_fc.mean():.0f}% coverage")
    with c3:
        mae_val   = float(abs_err[has_fc].mean()) if has_fc.any() else None
        mae_color = GREEN if (mae_val or 99) < 20 else AMBER
        metric_card("MAE (live)", f"{mae_val:.2f}" if mae_val else "—", "EUR/MWh", mae_color)
    with c4:
        if has_fc.any():
            cov = float(((actual[has_fc] >= p05[has_fc]) & (actual[has_fc] <= p95[has_fc])).mean() * 100)
            metric_card("PI Coverage", f"{cov:.1f}%", "q5–q95", GREEN if cov >= 85 else AMBER)
        else:
            metric_card("PI Coverage", "—", "no forecast data")

    st.markdown("---")

    # Main chart: actual vs live forecast
    fig = go.Figure()
    if has_fc.any():
        fc_idx = bt_lgbm.index[bt_lgbm["lgbm_q50"].notna()]
        fig.add_trace(_band_trace(
            fc_idx, bt_lgbm.loc[fc_idx, "lgbm_q05"],
            bt_lgbm.loc[fc_idx, "lgbm_q95"], "90% band",
        ))
        fig.add_trace(go.Scatter(
            x=fc_idx, y=bt_lgbm.loc[fc_idx, "lgbm_q50"],
            name="LightGBM p50 (live)", line=dict(color=BLUE, width=1.8),
        ))
    if bt_lear is not None:
        fig.add_trace(go.Scatter(
            x=bt_lear.index, y=bt_lear["lear_q50"],
            name="LEAR p50 (live)", line=dict(color=AMBER, width=1.5, dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        name="Actual", line=dict(color=ACTUAL, width=1.5),
    ))
    apply_layout(fig, "Actual vs Live Forecast (re-prediction)")
    st.plotly_chart(fig, use_container_width=True)

    if has_fc.any():
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Daily MAE trend")
            daily_mae = abs_err[has_fc].resample("D").mean().dropna()
            fig2 = go.Figure(go.Scatter(
                x=daily_mae.index, y=daily_mae.values,
                fill="toself", fillcolor="rgba(37,99,235,0.1)",
                line=dict(color=BLUE, width=1.5),
            ))
            fig2.update_layout(
                yaxis_title="MAE (EUR/MWh)", template="plotly_white",
                height=260, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            st.subheader("Error distribution")
            errors = err[has_fc].dropna()
            fig3 = go.Figure(go.Histogram(
                x=errors, nbinsx=40, marker=dict(color=BLUE, opacity=0.7),
            ))
            fig3.add_vline(x=0, line_color=RED, line_dash="dash")
            fig3.update_layout(
                xaxis_title="Error (EUR/MWh)", yaxis_title="Count",
                template="plotly_white", height=260,
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📋 Detailed table"):
            tbl = pd.DataFrame({
                "Actual":       actual[has_fc].round(2),
                "Forecast p50": p50[has_fc].round(2),
                "Lower q05":    p05[has_fc].round(2),
                "Upper q95":    p95[has_fc].round(2),
                "Error":        err[has_fc].round(2),
                "Abs error":    abs_err[has_fc].round(2),
            })
            st.dataframe(tbl, use_container_width=True)
            csv = tbl.reset_index().to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv,
                               file_name=f"backtest_{from_date}_{to_date}.csv",
                               mime="text/csv")

    # ── Bitemporal section ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Bitemporal Backtest")
    st.markdown(
        "The live forecast above re-runs today's model on past data — the model "
        "was trained on data that *includes* the backtest period (subtle look-ahead). "
        "TimeDB lets you query what was **actually known at the time**."
    )

    bt_mode = st.radio(
        "Mode",
        ["As-of query (end_known)", "Day-ahead gate closure (read_relative)"],
        horizontal=True,
    )

    if bt_mode == "As-of query (end_known)":
        st.markdown(
            "**`td.read(..., end_known=T)`** returns only forecasts with "
            "`knowledge_time < T`. Move the date to see how many forecasts existed."
        )
        as_of_date = st.date_input(
            "View forecasts as of (knowledge_time cut-off)",
            value=date.today(),
            key="as_of_date",
        )
        as_of_dt = datetime(as_of_date.year, as_of_date.month, as_of_date.day,
                            23, 59, tzinfo=timezone.utc)
        as_of_lgbm, as_of_lear = load_forecasts_as_of(
            as_of_dt.isoformat(), str(from_date), str(to_date)
        )

        if as_of_lgbm is not None:
            n_stored   = int(as_of_lgbm["lgbm_q50"].notna().sum())
            n_total    = len(bt_df)
            pct        = 100 * n_stored / n_total if n_total else 0
            mae_stored = float(np.abs(
                actual.reindex(as_of_lgbm.index) - as_of_lgbm["lgbm_q50"]
            ).dropna().mean()) if n_stored else None

            ca, cb, cc = st.columns(3)
            with ca:
                metric_card("Hours with stored forecast",
                            f"{n_stored:,}", f"{pct:.0f}% of window", BLUE)
            with cb:
                metric_card("MAE (stored as-of)",
                            f"{mae_stored:.2f}" if mae_stored else "—",
                            "EUR/MWh", GREEN if (mae_stored or 99) < 20 else AMBER)
            with cc:
                delta = ((mae_stored or 0) - (mae_val or 0)) if (mae_stored and mae_val) else None
                col   = RED if (delta or 0) > 0 else GREEN
                metric_card("Δ MAE vs live",
                            f"{delta:+.2f}" if delta is not None else "—",
                            "stored − live", col)

            # Overlay chart
            fig_aof = go.Figure()
            fig_aof.add_trace(go.Scatter(
                x=actual.index, y=actual,
                name="Actual", line=dict(color=ACTUAL, width=1.5),
            ))
            if has_fc.any():
                fig_aof.add_trace(go.Scatter(
                    x=p50[has_fc].index, y=p50[has_fc],
                    name="Live re-prediction", line=dict(color=GRAY, width=1.2, dash="dot"),
                ))
            aof_valid = as_of_lgbm["lgbm_q50"].dropna()
            fig_aof.add_trace(go.Scatter(
                x=aof_valid.index, y=aof_valid,
                name=f"As-of {as_of_date} (TimeDB)", line=dict(color=BLUE, width=2),
            ))
            apply_layout(fig_aof,
                         f"As-of query: forecasts known before {as_of_date}")
            st.plotly_chart(fig_aof, use_container_width=True)
        else:
            st.warning(
                f"No forecasts found in TimeDB with `knowledge_time < {as_of_date}`. "
                f"Try a later date — the model was trained on **{date.today()}**."
            )
            st.info(
                "💡 This is the bitemporal guarantee in action: if you ask for forecasts "
                "that didn't exist yet, TimeDB returns nothing rather than silently "
                "returning stale or fabricated values."
            )

    else:  # Day-ahead gate closure
        st.markdown(
            "**`td.read_relative(days_ahead=1, time_of_day=time(12,0))`** returns only the "
            "forecast that existed at **noon D-1** (SE3 gate closure). This is what a "
            "trading desk would have seen — no future data, guaranteed by the query itself."
        )
        da_lgbm, da_lear = load_day_ahead_gate_closure(str(from_date), str(to_date))

        if da_lgbm is not None and da_lgbm.notna().sum() > 0:
            n_da    = int(da_lgbm.notna().sum())
            mae_da  = float(np.abs(actual.reindex(da_lgbm.index) - da_lgbm).dropna().mean())

            ca, cb, cc = st.columns(3)
            with ca:
                metric_card("Gate-closure hours", f"{n_da:,}",
                            "available at noon D-1", BLUE)
            with cb:
                metric_card("MAE (gate-closure)", f"{mae_da:.2f}", "EUR/MWh",
                            GREEN if mae_da < 20 else AMBER)
            with cc:
                delta = (mae_da - (mae_val or 0)) if mae_val else None
                metric_card("Δ MAE vs live",
                            f"{delta:+.2f}" if delta is not None else "—",
                            "gate-closure − live", RED if (delta or 0) > 0 else GREEN)

            fig_da = go.Figure()
            fig_da.add_trace(go.Scatter(
                x=actual.index, y=actual,
                name="Actual", line=dict(color=ACTUAL, width=1.5),
            ))
            if has_fc.any():
                fig_da.add_trace(go.Scatter(
                    x=p50[has_fc].index, y=p50[has_fc],
                    name="Live re-prediction", line=dict(color=GRAY, width=1.2, dash="dot"),
                ))
            da_valid = da_lgbm.dropna()
            fig_da.add_trace(go.Scatter(
                x=da_valid.index, y=da_valid,
                name="Gate-closure forecast (TimeDB)", line=dict(color=BLUE, width=2),
            ))
            if da_lear is not None:
                da_lear_valid = da_lear.dropna()
                fig_da.add_trace(go.Scatter(
                    x=da_lear_valid.index, y=da_lear_valid,
                    name="LEAR gate-closure", line=dict(color=AMBER, width=1.5, dash="dash"),
                ))
            apply_layout(fig_da,
                         "Gate-closure backtest — read_relative(days_ahead=1, noon D-1)")
            st.plotly_chart(fig_da, use_container_width=True)
        else:
            st.warning(
                "No gate-closure forecasts found. This means the model hadn't been "
                "trained before noon D-1 for any day in this window."
            )
            st.info(
                "💡 `read_relative()` enforces `knowledge_time ≤ noon D-1` at the SQL level. "
                "There is no way to accidentally include a forecast that didn't exist yet."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_diag:
    st.title("🔬 Model Diagnostics")

    st.subheader("LightGBM — Feature importance")
    if has_models:
        try:
            fi = lgbm.feature_importance(top_n=15)
            fi_fig = go.Figure(go.Bar(
                x=fi["mean"], y=fi.index,
                orientation="h", marker_color=BLUE,
            ))
            fi_fig.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Mean importance (gain)",
                template="plotly_white",
                height=450, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fi_fig, use_container_width=True)
        except FileNotFoundError:
            st.info("Train models to see feature importance.")
    else:
        st.info("Run `python -m ml.train` to see feature importance.")

    st.markdown("---")
    st.subheader("Reliability diagram (calibration)")
    st.caption(
        "A perfectly calibrated model sits on the diagonal. "
        "Below → overconfident. Above → underconfident."
    )

    if has_prices and has_models:
        from ml.evaluate import _reliability_data
        forecasts_r = {}
        if lgbm_pred is not None:
            forecasts_r["lgbm"] = lgbm_pred
        if lear_pred is not None:
            forecasts_r["lear"] = lear_pred

        actuals_s  = df["price"].dropna()
        aligned_f  = {k: v.reindex(actuals_s.index) for k, v in forecasts_r.items()}
        rel        = _reliability_data(actuals_s, aligned_f)

        rel_fig = go.Figure()
        rel_fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color=GRAY, dash="dash"), name="Perfect calibration",
        ))
        for model, color in [("lgbm", BLUE), ("lear", AMBER)]:
            sub = rel[rel["model"] == model]
            if sub.empty:
                continue
            rel_fig.add_trace(go.Scatter(
                x=sub["nominal"], y=sub["observed"],
                mode="lines+markers", name=model.upper(),
                line=dict(color=color, width=2), marker=dict(size=8),
            ))
        rel_fig.update_layout(
            xaxis_title="Nominal quantile level",
            yaxis_title="Observed coverage",
            template="plotly_white", height=400,
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(tickformat=".0%"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(rel_fig, use_container_width=True)
    else:
        st.info("Reliability diagram requires price data and trained models.")

    st.markdown("---")
    st.subheader("Price spike analysis (> 100 EUR/MWh)")

    if has_prices:
        spike_mask = df["price"] > 100
        n_spikes   = int(spike_mask.sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Spike hours", str(n_spikes),
                        f"in last {days_back} days", RED if n_spikes else GREEN)
        if has_models and lgbm_pred is not None and n_spikes > 0:
            spike_err = np.abs(
                df.loc[spike_mask, "price"] - lgbm_pred.loc[spike_mask, "lgbm_q50"]
            ).dropna()
            all_err = np.abs((df["price"] - lgbm_pred["lgbm_q50"]).dropna())
            with c2:
                metric_card("Spike MAE — LightGBM", f"{spike_err.mean():.2f}", "EUR/MWh", RED)
            with c3:
                metric_card("Overall MAE", f"{all_err.mean():.2f}", "EUR/MWh", BLUE)

        if n_spikes > 0:
            sp_fig = go.Figure()
            sp_fig.add_trace(go.Scatter(
                x=df.index, y=df["price"],
                name="Price", line=dict(color=GRAY, width=0.8), opacity=0.6,
            ))
            sp_fig.add_trace(go.Scatter(
                x=df[spike_mask].index, y=df.loc[spike_mask, "price"],
                mode="markers", name="Spike",
                marker=dict(color=RED, size=8, symbol="x"),
            ))
            apply_layout(sp_fig, "Price spikes highlighted")
            st.plotly_chart(sp_fig, use_container_width=True)
    else:
        st.info("Spike analysis requires price data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    st.title("🗃️ Data Explorer")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        de_from = st.date_input("From", value=date.today() - timedelta(days=7), key="de_from")
    with col2:
        de_to   = st.date_input("To",   value=date.today(), key="de_to")
    with col3:
        dataset = st.selectbox("Dataset", ["Prices", "Weather"])

    if de_from >= de_to:
        st.error("From date must be before To date.")
        st.stop()

    window = df[(df.index.date >= de_from) & (df.index.date <= de_to)]
    st.caption(f"**{len(window):,} hours** · {de_from} → {de_to}")

    if dataset == "Prices":
        if not has_prices:
            st.info("No price data yet — ENTSO-E sync pending.")
            st.stop()
        p = window["price"].dropna()
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Mean", f"{p.mean():.2f}", "EUR/MWh")
        with c2: metric_card("Max",  f"{p.max():.2f}",  "EUR/MWh", RED)
        with c3: metric_card("Min",  f"{p.min():.2f}",  "EUR/MWh", GREEN)
        with c4: metric_card("Std",  f"{p.std():.2f}",  "EUR/MWh", GRAY)
        fig = go.Figure(go.Scatter(
            x=p.index, y=p, name="Price", fill="tozeroy",
            line=dict(color=BLUE, width=1.2),
            fillcolor="rgba(37,99,235,0.08)",
        ))
        apply_layout(fig, "SE3 Day-Ahead Prices")
        st.plotly_chart(fig, use_container_width=True)

    elif dataset == "Weather":
        tab_w, tab_t, tab_s = st.tabs(["Wind", "Temperature", "Solar"])
        with tab_w:
            fig = go.Figure(go.Scatter(
                x=window.index, y=window["wind_speed"],
                name="Wind speed (10 m)", line=dict(color=BLUE, width=1.5),
            ))
            apply_layout(fig, "Wind Speed (10 m)", "m/s")
            st.plotly_chart(fig, use_container_width=True)
        with tab_t:
            fig = go.Figure(go.Scatter(
                x=window.index, y=window["temperature"],
                name="Temperature", fill="tozeroy",
                line=dict(color=AMBER, width=1.5),
                fillcolor="rgba(245,158,11,0.08)",
            ))
            apply_layout(fig, "Temperature — Stockholm", "°C")
            st.plotly_chart(fig, use_container_width=True)
        with tab_s:
            fig = go.Figure(go.Scatter(
                x=window.index, y=window["irradiance"],
                name="Solar irradiance", fill="tozeroy",
                line=dict(color=AMBER, width=1.5),
                fillcolor="rgba(245,158,11,0.08)",
            ))
            apply_layout(fig, "Shortwave Solar Irradiance", "W/m²")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw data table"):
        st.dataframe(window.round(2), use_container_width=True)
        csv = window.round(2).reset_index().to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV", data=csv,
            file_name=f"se3_{dataset.lower()}_{de_from}_{de_to}.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.title("ℹ️ About this project")
    st.markdown("""
## SE3 Electricity Price Forecast — v2

Forecasts **day-ahead electricity prices** for the **SE3 bidding zone** (Sweden) using a
production-grade ML pipeline backed by a bitemporal time-series store.

### Models

| Model | Description |
|---|---|
| **LightGBM** | Gradient-boosted trees — three separate models for p05/p50/p95 quantiles |
| **LEAR** | LASSO Estimated AutoRegressive (Lago et al. 2021) — one model per hour-of-day |

### Features
Price lags (23/24/25/48/72/168/336 h), rolling means, cyclically encoded calendar
variables, Swedish public holidays, temperature, wind speed, solar irradiance,
and interaction terms (hour×month, weekend×hour, temperature×hour, temperature×wind).

### Evaluation
Walk-forward validation with CRPS (primary), MAE, RMSE, MAPE,
interval coverage, spike MAE, and calibration via reliability diagrams.

### Bitemporal storage — TimeDB
**TimeDB** (rebase.energy) — ClickHouse-backed bitemporal store. Every observation
carries both a *valid_time* (when the price occurs) and a *knowledge_time* (when it
was known), enabling rigorous point-in-time backtesting:

- `td.read(..., end_known=T)` — as-of query: "what did we know before time T?"
- `td.read_relative(days_ahead=1, time_of_day=time(12,0))` — gate-closure backtest:
  "what forecast was available at noon D-1, the SE3 auction gate closure?"

### Data sources
- **ENTSO-E Transparency Platform** — SE3 day-ahead auction prices
- **Open-Meteo** — historical and forecast weather for Stockholm (59.33°N, 18.07°E)

### References
Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021).
*Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms,
best practices and an open-access benchmark.* Applied Energy, 293, 116983.
""")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:13px;padding:8px'>"
    "Built by <b>Arnob Mukherjee</b> &nbsp;·&nbsp; "
    "Data: <a href='https://transparency.entsoe.eu' target='_blank' "
    "style='color:#2563eb'>ENTSO-E</a> + "
    "<a href='https://open-meteo.com' target='_blank' "
    "style='color:#2563eb'>Open-Meteo</a> &nbsp;·&nbsp; "
    "Storage: <a href='https://github.com/rebase-energy/timedb' target='_blank' "
    "style='color:#2563eb'>TimeDB</a>"
    "</div>",
    unsafe_allow_html=True,
)
