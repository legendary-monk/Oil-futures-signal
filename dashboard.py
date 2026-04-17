"""Streamlit dashboard for oil signal performance analytics."""

from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Oil Signal Dashboard", layout="wide")
st.title("🛢 Oil Signal Dashboard")

csv_path = Path(config.PREDICTIONS_FILE)
if not csv_path.exists():
    st.warning(f"Predictions file not found: {csv_path}")
    st.stop()


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


df = pd.read_csv(csv_path)
if df.empty:
    st.warning("Predictions CSV is empty.")
    st.stop()

if "date" not in df.columns:
    st.error("CSV missing required 'date' column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for col in ["change_pct_1d", "change_pct_5d", "confidence", "polymarket_score", "sentiment_score", "trend_score", "macro_signal"]:
    if col in df.columns:
        df[col] = _to_num(df[col])

# Outcome helpers
for horizon, col in [("1d", "outcome_1d"), ("5d", "outcome_5d")]:
    hit_col = f"hit_{horizon}"
    if col in df.columns:
        df[hit_col] = (df[col] == "CORRECT").astype("float")
    else:
        df[hit_col] = pd.NA

# Sidebar filters
st.sidebar.header("Filters")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

opec_mode = st.sidebar.selectbox(
    "OPEC window",
    ["All", "Exclude OPEC uncertainty", "Only OPEC uncertainty"],
    index=0,
)

fdf = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()
if "opec_uncertainty" in fdf.columns:
    if opec_mode == "Exclude OPEC uncertainty":
        fdf = fdf[fdf["opec_uncertainty"].astype(str) == "0"]
    elif opec_mode == "Only OPEC uncertainty":
        fdf = fdf[fdf["opec_uncertainty"].astype(str) == "1"]

if fdf.empty:
    st.warning("No rows after filters.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
resolved_1d = fdf[fdf.get("outcome_1d", "").isin(["CORRECT", "INCORRECT"])] if "outcome_1d" in fdf.columns else pd.DataFrame()
resolved_5d = fdf[fdf.get("outcome_5d", "").isin(["CORRECT", "INCORRECT"])] if "outcome_5d" in fdf.columns else pd.DataFrame()

acc_1d = (resolved_1d["outcome_1d"].eq("CORRECT").mean() * 100) if not resolved_1d.empty else float("nan")
acc_5d = (resolved_5d["outcome_5d"].eq("CORRECT").mean() * 100) if not resolved_5d.empty else float("nan")

c1.metric("Rows", len(fdf))
c2.metric("1D accuracy", f"{acc_1d:.1f}%" if pd.notna(acc_1d) else "N/A")
c3.metric("5D accuracy", f"{acc_5d:.1f}%" if pd.notna(acc_5d) else "N/A")
c4.metric("Directional signals", int(fdf["signal"].isin(["BULLISH", "BEARISH"]).sum()) if "signal" in fdf.columns else 0)

st.subheader("Rolling directional accuracy")
roll = fdf.copy()
if "hit_1d" in roll.columns:
    roll["rolling_1d_acc_pct"] = roll["hit_1d"].rolling(20, min_periods=5).mean() * 100
if "hit_5d" in roll.columns:
    roll["rolling_5d_acc_pct"] = roll["hit_5d"].rolling(20, min_periods=5).mean() * 100

plot_cols = [c for c in ["rolling_1d_acc_pct", "rolling_5d_acc_pct"] if c in roll.columns]
if plot_cols:
    st.line_chart(roll.set_index("date")[plot_cols])

st.subheader("Accuracy by confidence band")
if "confidence" in fdf.columns and "outcome_1d" in fdf.columns:
    directional = fdf[fdf["signal"].isin(["BULLISH", "BEARISH"])].copy()
    directional = directional[directional["outcome_1d"].isin(["CORRECT", "INCORRECT"])]
    if not directional.empty:
        directional["band"] = pd.cut(
            directional["confidence"],
            bins=[-1, 0.45, 0.65, 1.0],
            labels=["Low (<45%)", "Mid (45-65%)", "High (>=65%)"],
        )
        band_acc = directional.groupby("band", observed=False)["outcome_1d"].apply(lambda s: (s == "CORRECT").mean() * 100)
        st.bar_chart(band_acc)

st.subheader("Signal distribution")
if "signal" in fdf.columns:
    st.bar_chart(fdf["signal"].value_counts())

st.subheader("Factor vs 1D return correlation")
factor_cols = [c for c in ["polymarket_score", "sentiment_score", "trend_score", "macro_signal"] if c in fdf.columns]
if "change_pct_1d" in fdf.columns and factor_cols:
    corr_df = fdf[factor_cols + ["change_pct_1d"]].corr(numeric_only=True)
    st.dataframe(corr_df[["change_pct_1d"]].sort_values("change_pct_1d", ascending=False))

st.subheader("Recent rows")
st.dataframe(fdf.sort_values("date", ascending=False).head(30), use_container_width=True)
