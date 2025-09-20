# graphs.py
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Wealth Genie — Graphs", page_icon="📈")
st.title("Graphs")

# --- Pull results from session (set by main.py) ---
stock_res  = st.session_state.get("stock_result",  {"Top 5 Stocks Return": {}})
gold_res   = st.session_state.get("gold_result",   {"Top 5 Gold Return":   {}})
crypto_res = st.session_state.get("crypto_result", {"Top 5 Crypto Return": {}})

gold_weight   = st.session_state.get("gold_weight")
stock_weight  = st.session_state.get("stock_weight")
crypto_weight = st.session_state.get("crypto_weight")

# Guard-rails (fix: check correct keys)
have_any = any([
    bool(stock_res.get("Top 5 Stocks Return")),
    bool(gold_res.get("Top 5 Gold Return")),
    bool(crypto_res.get("Top 5 Crypto Return")),
])

if not have_any or gold_weight is None or stock_weight is None or crypto_weight is None:
    st.info("No results to show yet. Go to **Quiz** (main page), submit your answers, and run the forecasts first.")
    st.stop()

# ---------- Helpers ----------
def _top_series(d: dict) -> pd.Series:
    if not isinstance(d, dict) or not d:
        return pd.Series(dtype="float64")
    return pd.Series(d, dtype="float64").sort_values(ascending=False)

def bar_with_axes(df: pd.DataFrame, x_col: str, y_col: str,
                  x_title: str, y_title: str,
                  y_format: str = ".2f", y_domain=None, height: int = 280):
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=x_title, sort=None),
        y=alt.Y(f"{y_col}:Q", title=y_title,
                scale=alt.Scale(domain=y_domain) if y_domain is not None else alt.Undefined)
    )
    labels = alt.Chart(df).mark_text(align="center", baseline="bottom", dy=-3).encode(
        x=f"{x_col}:N",
        y=f"{y_col}:Q",
        text=alt.Text(f"{y_col}:Q", format=y_format)
    )
    return (bars + labels).properties(height=height)

# ---------- Allocation ----------
st.subheader("Portfolio Allocation (from your risk score)")

alloc_df = pd.DataFrame(
    {"Asset": ["Gold", "Stocks", "Crypto"],
     "Weight": [gold_weight, stock_weight, crypto_weight]}
)

alloc_chart = alt.Chart(alloc_df).mark_bar().encode(
    x=alt.X("Asset:N", title="Asset class"),
    y=alt.Y("Weight:Q", title="Portfolio weight", scale=alt.Scale(domain=[0, 1]))
).properties(height=280)

alloc_labels = alt.Chart(alloc_df).mark_text(align="center", baseline="bottom", dy=-3).encode(
    x="Asset:N",
    y="Weight:Q",
    text=alt.Text("Weight:Q", format=".0%")
)

st.altair_chart(alloc_chart + alloc_labels, use_container_width=True)

# ---------- Top per category ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Top 5 Stocks (predicted % next N days)")
    s = _top_series(stock_res.get("Top 5 Stocks Return", {}))
    if s.empty:
        st.caption("No stock predictions available.")
    else:
        df = s.reset_index()
        df.columns = ["Ticker", "Return"]
        chart = bar_with_axes(
            df, "Ticker", "Return",
            x_title="Ticker", y_title="Predicted return (%) — next N days",
            y_format=".2f"
        )
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Top 3 Gold (predicted % next N days)")
    s = _top_series(gold_res.get("Top 5 Gold Return", {})).head(3)
    if s.empty:
        st.caption("No gold predictions available.")
    else:
        df = s.reset_index()
        df.columns = ["Ticker", "Return"]
        chart = bar_with_axes(
            df, "Ticker", "Return",
            x_title="Ticker", y_title="Predicted return (%) — next N days",
            y_format=".2f"
        )
        st.altair_chart(chart, use_container_width=True)

with col3:
    st.subheader("Top 5 Crypto (predicted % next N days)")
    s = _top_series(crypto_res.get("Top 5 Crypto Return", {}))
    if s.empty:
        st.caption("No crypto predictions available.")
    else:
        df = s.reset_index()
        df.columns = ["Ticker", "Return"]
        chart = bar_with_axes(
            df, "Ticker", "Return",
            x_title="Ticker", y_title="Predicted return (%) — next N days",
            y_format=".2f"
        )
        st.altair_chart(chart, use_container_width=True)

st.caption("Educational visualizations only — not investment advice.")
