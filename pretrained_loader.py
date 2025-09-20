# pretrained_loader.py
import os, glob, math, json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


def _split_patterns(s: str) -> List[str]:
    """
    Split a user string like:
      'artifacts*; C:/work/models ;D:/bags/*.dir'
    into clean glob patterns.
    """
    if not s:
        return []
    raw = [p.strip().replace("\\", "/") for p in re_split_multi(s)]
    return [p for p in raw if p]


def re_split_multi(s: str) -> List[str]:
    # split on ; , whitespace newlines
    import re
    return re.split(r"[;,\n]+", s)


def _glob_first(patterns: List[str]) -> Optional[str]:
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            hits.sort(key=lambda x: (len(x), x.lower()))
            return hits[0]
    return None


def find_model_and_scaler(artifacts_patterns: str, ticker: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (model_path, scaler_path, meta_path) or (None, None, None).
    `artifacts_patterns` can be a single path or multiple paths/globs separated by ; or , or newlines.
    Examples:
      'artifacts'                    # single dir
      'artifacts*'                   # matches artifacts-AAPL, artifacts-MSFT, ...
      'C:/proj/artifacts*;D:/extra'  # multiple roots
    """
    pats = _split_patterns(artifacts_patterns)
    if not pats:
        pats = ["artifacts*", "artifacts", "."]  # sensible defaults

    # Compose search patterns across all roots
    model_candidates, scaler_candidates, meta_candidates = [], [], []
    for root in pats:
        root = root.rstrip("/")

        # common layouts:
        # - root/**/*{ticker}*.keras / .h5
        # - root-* (when root itself is 'artifacts*')
        model_candidates += [
            f"{root}/**/*{ticker}*.keras",
            f"{root}/**/*{ticker}*.h5",
        ]
        scaler_candidates += [
            f"{root}/**/*{ticker}*yscaler*.pkl",
            f"{root}/**/*{ticker}*xscaler*.pkl",
            f"{root}/**/*{ticker}*scaler*.pkl",
        ]
        meta_candidates += [
            f"{root}/**/*{ticker}*meta*.json",
            f"{root}/**/{ticker}.json",
            f"{root}/**/*{ticker}*.json",
        ]

        # Also try a folder literally named artifacts-{ticker}
        base = root.rstrip("*")
        if base:
            model_candidates += [
                f"{base}-{ticker}/**/*.keras",
                f"{base}-{ticker}/**/*.h5",
            ]
            scaler_candidates += [
                f"{base}-{ticker}/**/*scaler*.pkl",
                f"{base}-{ticker}/**/*yscaler*.pkl",
                f"{base}-{ticker}/**/*xscaler*.pkl",
            ]
            meta_candidates += [
                f"{base}-{ticker}/**/*meta*.json",
                f"{base}-{ticker}/**/*.json",
            ]

    model_path = _glob_first(model_candidates)
    scaler_path = _glob_first(scaler_candidates)
    meta_path   = _glob_first(meta_candidates)
    return model_path, scaler_path, meta_path


def fetch_close(ticker: str, period="3y", interval="1d"):
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No data for {ticker}")
    px = df["Close"].astype("float32").to_numpy()
    return px, df.index.to_list()


def make_returns(px: np.ndarray) -> np.ndarray:
    r = np.zeros_like(px, dtype=np.float32)
    r[1:] = np.log(px[1:] / np.maximum(1e-6, px[:-1]))
    return r


def adaptive_lookback(px_len: int, requested: int, min_after_lb: int = 30) -> int:
    max_lb = max(10, px_len - min_after_lb)
    return int(max(10, min(requested, max_lb)))


def load_pretrained(ticker: str, artifacts_patterns: str):
    """
    Load a compiled=False model (inference), optional scaler, and optional meta JSON.
    Returns (model, scaler_or_None, meta_dict, reason_if_skipped).
    """
    mpath, spath, jpath = find_model_and_scaler(artifacts_patterns, ticker)
    if not mpath:
        return None, None, {}, f"no model file found for {ticker} (patterns={artifacts_patterns})"

    try:
        model = load_model(mpath, compile=False)
    except Exception as e:
        return None, None, {}, f"failed to load model: {e}"

    scaler = None
    if spath:
        try:
            scaler = joblib.load(spath)
        except Exception as e:
            return None, None, {}, f"failed to load scaler: {e}"

    meta = {}
    if jpath:
        try:
            with open(jpath, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return model, scaler, meta, ""


def predict_next_prices_from_returns_model(
    ticker: str,
    model,
    scaler,             # may be None → we’ll fit a fallback StandardScaler on recent returns
    lookback: int,
    horizon: int,
    period: str = "3y",
    interval: str = "1d",
):
    try:
        px, _ = fetch_close(ticker, period=period, interval=interval)
    except Exception as e:
        return None, f"data error: {e}"

    lookback = adaptive_lookback(len(px), lookback, min_after_lb=30)
    if len(px) < lookback + 2:
        return None, f"not enough data (len={len(px)}, lb={lookback})"

    r = make_returns(px)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(r.reshape(-1, 1))

    win_sc = scaler.transform(r[-lookback:].reshape(-1, 1)).reshape(1, lookback, 1)

    fut_prices = []
    carry = float(px[-1])

    for _ in range(int(horizon)):
        nxt_sc = float(model.predict(win_sc, verbose=0)[0][0])
        nxt_r  = float(scaler.inverse_transform([[nxt_sc]])[0][0])
        carry  = float(carry * math.exp(nxt_r))
        fut_prices.append(carry)
        win_sc = np.concatenate([win_sc[:, 1:, :], np.array([[[nxt_sc]]], dtype=np.float32)], axis=1)

    return fut_prices, None
