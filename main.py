# main.py
import os
import time
import glob
import math
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Wealth Genie", page_icon="💹")
st.title("Quiz")

# -------------------------------
# Session state init
# -------------------------------
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# -------------------------------
# Tickers
# -------------------------------
STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "V", "XOM", "KO"]
GOLD_TICKERS  = ["GLD", "IAU", "GDX"]
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "AVAX-USD"]
ALL_TICKERS = STOCK_TICKERS + GOLD_TICKERS + CRYPTO_TICKERS

# =========================
# Quiz UI
# =========================
questions = [f"question{i}" for i in range(1, 14)]
money_invested1 = st.number_input("Amount to be Invested", min_value=0, key="money_invested1")

def radio(q, prompt, options):
    return st.radio(prompt, options, index=None, key=q)

question1  = radio("question1",  "1. In general, how would your best friend describe you as a risk taker?", [
    "a. A real gambler","b. Willing to take risks after completing adequate research","c. Cautious","d. A real risk avoider"
])
question2  = radio("question2",  "2. You are on a TV game show and can choose one of the following. Which would you take?", [
    "a. $1,000 in cash","b. A 50% chance at winning $5,000","c. A 25% chance at winning $10,000","d. A 5% chance at winning $100,000"
])
question3  = radio("question3",  "3. You finished saving for a once-in-a-lifetime vacation, then you lose your job. You would:", [
    "a. Cancel the vacation","b. Take a much more modest vacation","c. Go as scheduled","d. Extend your vacation (last chance!)"
])
question4  = radio("question4",  "4. If you unexpectedly received $20,000 to invest, what would you do?", [
    "a. Bank/MM/CD","b. High-quality bonds/bond funds","c. Stocks/stock funds"
])
question5  = radio("question5",  "5. How comfortable are you investing in stocks?", [
    "a. Not at all comfortable","b. Somewhat comfortable","c. Very comfortable"
])
question6  = radio("question6",  "6. The word “risk” makes you think of:", [
    "a. Loss","b. Uncertainty","c. Opportunity","d. Thrill"
])
question7  = radio("question7",  "7. Your assets are in high-interest gov bonds, experts warn prices may fall; what do you do?", [
    "a. Hold the bonds","b. Sell half → MM + hard assets","c. Sell all → hard assets","d. Sell all + borrow more for hard assets"
])
question8  = radio("question8",  "8. Pick best/worst payoff combo:", [
    "a. +200 / 0","b. +800 / -200","c. +2600 / -800","d. +4800 / -2400"
])
question9  = radio("question9",  "9. You’re given $1,000. Choose:", [
    "a. Sure gain of $500","b. 50% to gain $1,000 / 50% gain $0"
])
question10 = radio("question10","10. You’re given $2,000. Choose:", [
    "a. Sure loss of $500","b. 50% to lose $1,000 / 50% lose $0"
])
question11 = radio("question11","11. Inherit $100k, must invest all in one:", [
    "a. Savings/MM","b. Balanced fund","c. 15 stocks","d. Commodities"
])
question12 = radio("question12","12. Invest $20,000, pick mix:", [
    "a. 60% low / 30% mid / 10% high","b. 30% low / 40% mid / 30% high","c. 10% low / 40% mid / 50% high"
])
question13 = radio("question13","13. Friend’s risky gold mine, 20% success (50–100×). Invest:", [
    "a. Nothing","b. One month’s salary","c. Three months’ salary","d. Six months’ salary"
])

def submit():
    filled = money_invested1 > 0 and all(st.session_state.get(k) for k in questions)
    if not filled:
        st.error("Please answer all questions and enter investment amount > 0.")
        return
    for q in questions:
        st.session_state.answers[q] = st.session_state[q]
    st.session_state.submitted = True
    st.session_state.answers["money_invested1"] = money_invested1

st.button("Submit Answers", on_click=submit)

# =========================
# Risk Scoring
# =========================
def _score(opt, mapping):
    for prefix, val in mapping:
        if opt and opt.startswith(prefix):
            return val
    return 0

def calculate_risk_score(ans):
    s = 0
    s += _score(ans["question1"],  [("a.",4),("b.",3),("c.",2),("d.",1)])
    s += _score(ans["question2"],  [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question3"],  [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question4"],  [("a.",1),("b.",2),("c.",3)])
    s += _score(ans["question5"],  [("a.",1),("b.",2),("c.",3)])
    s += _score(ans["question6"],  [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question7"],  [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question8"],  [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question9"],  [("a.",1),("b.",3)])
    s += _score(ans["question10"], [("a.",1),("b.",3)])
    s += _score(ans["question11"], [("a.",1),("b.",2),("c.",3),("d.",4)])
    s += _score(ans["question12"], [("a.",1),("b.",2),("c.",3)])
    s += _score(ans["question13"], [("a.",1),("b.",2),("c.",3),("d.",4)])
    return s

def risk_score_grable(score):
    if 0 <= score <= 18:
        return (0.2/18)*score
    if 18 <= score <= 22:
        return 0.2 + ((0.4-0.2)/(22-18))*(score-18)
    if 22 <= score <= 28:
        return 0.4 + ((0.6-0.4)/(28-22))*(score-22)
    if 28 < score <= 32:
        return 0.6 + ((0.8-0.6)/(32-28))*(score-28)
    if 32 < score <= 45:
        return 0.8 + ((1.0-0.8)/(45-32))*(score-32)
    return 1.0

# =========================
# Lookback CSV loader (OFFLINE FIRST)
# =========================
def _env_or_default_lookback_root() -> str:
    return os.environ.get("LOOKBACK_ROOT", "lookback")

def _find_lookback_csv(ticker: str) -> Optional[str]:
    root = _env_or_default_lookback_root()
    patterns = [
        os.path.join(root, f"artifacts-{ticker}", "inference_input_lookback*.csv"),
        os.path.join(root, f"artificats-{ticker}", "inference_input_lookback*.csv"),  # typo-friendly
        os.path.join("lookback_inputs", "lookback_run_*", f"artifacts-{ticker}", "inference_input_lookback*.csv"),
        os.path.join("lookback_inputs", f"artifacts-{ticker}", "inference_input_lookback*.csv"),
    ]
    hits: List[str] = []
    for p in patterns:
        hits.extend(glob.glob(p))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]

def load_lookback_window_closes(ticker: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    path = _find_lookback_csv(ticker)
    if not path or not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        if "Close" not in df.columns:
            return None, None
        closes = df["Close"].astype("float32").to_numpy()
        lb = max(0, len(closes) - 1)
        meta = {"src": "lookback_csv", "path": path, "lookback": lb}
        return closes, meta
    except Exception:
        return None, None

# =========================
# Model loaders (cached)
# =========================
def _find_artifact_paths(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    model_globs = [
        f"artifacts-{ticker}/*.keras",
        f"artifacts-{ticker}/*.h5",
        f"artifacts/{ticker}/*.keras",
        f"artifacts/{ticker}/*.h5",
    ]
    scaler_globs = [
        f"artifacts-{ticker}/*scaler*.joblib",
        f"artifacts-{ticker}/*scaler*.pkl",
        f"artifacts/{ticker}/*scaler*.joblib",
        f"artifacts/{ticker}/*scaler*.pkl",
        f"artifacts-{ticker}/*yscaler*.pkl",
        f"artifacts/{ticker}/*yscaler*.pkl",
    ]
    model_path = None
    scaler_path = None

    for g in model_globs:
        files = glob.glob(g)
        if files:
            files.sort(key=len)
            model_path = files[0]
            break
    for g in scaler_globs:
        files = glob.glob(g)
        if files:
            files.sort(key=len)
            scaler_path = files[0]
            break
    return model_path, scaler_path

@st.cache_resource(show_spinner=False)
def load_pretrained_bundle(ticker: str):
    model_path, scaler_path = _find_artifact_paths(ticker)
    if not model_path or not os.path.exists(model_path):
        return None, None, f"model not found in artifacts for {ticker}"
    if not scaler_path or not os.path.exists(scaler_path):
        return None, None, f"scaler not found in artifacts for {ticker}"

    try:
        mdl = load_model(model_path, compile=False)
    except Exception as e:
        return None, None, f"load model error: {e}"

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        return None, None, f"load scaler error: {e}"

    return mdl, scaler, None

# =========================
# Forecast helpers
# =========================
def make_log_returns(px: np.ndarray) -> np.ndarray:
    r = np.zeros_like(px, dtype=np.float32)
    r[1:] = np.log(px[1:] / np.maximum(1e-6, px[:-1]))
    return r

def horizon_return_pct_from_model(
    ticker: str,
    model,
    scaler,
    px: np.ndarray,
    lookback_from_model: Optional[int],
    horizon: int
) -> Tuple[Optional[float], Optional[str]]:
    try:
        lb = int(lookback_from_model or model.input_shape[1])
    except Exception:
        return None, "cannot infer lookback from model"

    if len(px) < lb + 1:
        return None, f"not enough data (need {lb+1}, got {len(px)})"

    r = make_log_returns(px)
    win_sc = scaler.transform(r[-lb:].reshape(-1, 1)).reshape(1, lb, 1)

    sum_r = 0.0
    for _ in range(int(horizon)):
        nxt_sc = float(model.predict(win_sc, verbose=0)[0][0])
        nxt_r  = float(scaler.inverse_transform([[nxt_sc]])[0][0])
        sum_r += nxt_r
        win_sc = np.concatenate([win_sc[:, 1:, :], np.array([[[nxt_sc]]], dtype=np.float32)], axis=1)

    pct = (math.exp(sum_r) - 1.0) * 100.0
    return pct, None

# =========================
# On submit → use pre-trained models + lookback CSVs
# =========================
if st.session_state.submitted:
    # 1) Risk → weights
    score_raw = calculate_risk_score(st.session_state.answers)
    risk_norm = risk_score_grable(score_raw)
    gold_w  = float(max(0.0, 0.6 - 0.6*risk_norm))
    stock_w = float(0.3 + 0.4*risk_norm)
    crypto_w= float(1.0 - gold_w - stock_w)
    crypto_w = max(0.0, crypto_w)

    st.session_state.gold_weight   = gold_w
    st.session_state.stock_weight  = stock_w
    st.session_state.crypto_weight = crypto_w

    st.success(f"Risk score: {score_raw} → normalized {risk_norm:.2f} | Weights → Gold {gold_w:.2f}, Stocks {stock_w:.2f}, Crypto {crypto_w:.2f}")

    # 2) Controls for forecaster (inference only)
    st.subheader("Load pre-trained LSTM models and forecast (offline-first)")
    horizon   = st.slider("Forecast horizon (days)", 1, 7, 3)
    per_cat   = st.slider("Max tickers per category (speed control)", 1, 10, 5)

    # Keep meta in session for other pages (hidden here)
    inference_meta: Dict[str, Dict[str, Any]] = {}

    def build_top5(tickers: List[str], label: str) -> Dict[str, float]:
        outs: List[Tuple[str, float]] = []
        to_eval = tickers[:per_cat]
        skipped: Dict[str, str] = {}

        for t in to_eval:
            with st.spinner(f"{t}: loading model & inputs…"):
                model, scaler, err = load_pretrained_bundle(t)
                if err:
                    skipped[t] = err
                    continue

                px_arr, meta = load_lookback_window_closes(t)
                if px_arr is None:
                    skipped[t] = "no lookback CSV found under lookback/*"
                    continue

                lb = None
                try:
                    lb = int(model.input_shape[1])
                except Exception:
                    lb = None

                pct, err2 = horizon_return_pct_from_model(
                    ticker=t, model=model, scaler=scaler, px=px_arr,
                    lookback_from_model=lb, horizon=horizon
                )
                if err2:
                    skipped[t] = err2
                    continue

                inference_meta[t] = meta or {"src":"unknown"}
                if pct is not None and np.isfinite(pct):
                    outs.append((t, round(float(pct), 2)))

        outs.sort(key=lambda x: x[1], reverse=True)
        if skipped:
            st.info("Skipped: " + "; ".join(f"{k}: {v}" for k,v in skipped.items()))
        return dict(outs[:5])

    dict_stock_top  = build_top5(STOCK_TICKERS, "stocks")
    dict_gold_top   = build_top5(GOLD_TICKERS,  "gold")
    dict_crypto_top = build_top5(CRYPTO_TICKERS,"crypto")

    # Store for other pages (graphs/chatbot) + meta (no on-page preview/tables)
    st.session_state.stock_result  = {"Top 5 Stocks Return": dict_stock_top}
    st.session_state.gold_result   = {"Top 5 Gold Return": dict_gold_top}
    st.session_state.crypto_result = {"Top 5 Crypto Return": dict_crypto_top}
    st.session_state.inference_meta = inference_meta

    st.success("Predictions ready. Open **Graphs** or **Chatbot** pages from the sidebar.")

else:
    st.info("Fill the quiz and press **Submit Answers** to see forecasts from your pre-trained models (using lookback CSVs).")

st.caption("This app is for education only and **not** financial advice.")
