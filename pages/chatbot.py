# chatbot.py
import os
import json
import numpy as np
import streamlit as st

# ---- Optional FAISS; fallback to pure NumPy if unavailable ----
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Wealth Genie", page_icon="💹")
st.title("Welcome to Wealth Genie!")

# =========================================================
#                 GROQ API KEY (hardcoded)
# =========================================================
# Put your actual key string here. No UI prompt will be shown.
GROQ_API_KEY = ""  # <-- your key

def _get_groq_key():
    # Use the hardcoded key if set; otherwise try env/secrets (no UI).
    if GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_XXX"):
        return GROQ_API_KEY
    try:
        k = st.secrets.get("GROQ_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        k = None
    return k or os.environ.get("GROQ_API_KEY")

# =========================================================
#              Session data & safe getters
# =========================================================
def _safe_get(name, default=None):
    return st.session_state.get(name, default)

crypto_weight = _safe_get("crypto_weight", 0.0)
gold_weight   = _safe_get("gold_weight",   0.0)
stock_weight  = _safe_get("stock_weight",  0.0)

stock_res  = _safe_get("stock_result",  {"Top 5 Stocks Return": {}})
gold_res   = _safe_get("gold_result",   {"Top 5 Gold Return":   {}})
crypto_res = _safe_get("crypto_result", {"Top 5 Crypto Return": {}})

# Optional: where inputs came from (set by main.py)
inference_meta = _safe_get("inference_meta", {})  # { TICKER: {lookback, src, path}, ... }

def _fmt_top5(d: dict, title: str) -> str:
    if not isinstance(d, dict) or not d:
        return f"{title}\nNone yet—run forecasts on the main page."
    rows = [f"- {k}: {v}%" for k, v in d.items()]
    return f"{title}\n" + "\n".join(rows)

# =========================================================
#                Build dynamic knowledge
# =========================================================
def base_chunks():
    return [
        "Stocks are ownership in a company; prices reflect expectations and performance.",
        "Cryptocurrencies are decentralized digital assets with high volatility.",
        "Gold is a commodity often used as an inflation hedge and safe haven.",
        "Volatility is the size of price swings over time.",
        "Returns measure profit or loss over a period.",
        "Momentum is the tendency to keep moving in the same direction.",
        "A moving average smooths price to reveal trend direction.",
        "Top crypto can surge on adoption/tech news; weak stocks often follow poor earnings or guidance.",
        "Gold tends to rise during inflation or periods of uncertainty.",
        "Modern Portfolio Theory balances risk and return via diversification and covariance.",
        "Risk scoring uses your 13-question quiz normalized to [0,1] to tune allocations.",
        "This app is educational only and not financial advice.",
    ]

def dynamic_chunks():
    chunks = [
        f"Current weights — Crypto: {crypto_weight:.2f}, Gold: {gold_weight:.2f}, Stocks: {stock_weight:.2f}.",
        _fmt_top5(gold_res.get("Top 5 Gold Return", {}),    "Top 5 Gold (predicted % returns):"),
        _fmt_top5(crypto_res.get("Top 5 Crypto Return", {}),"Top 5 Crypto (predicted % returns):"),
        _fmt_top5(stock_res.get("Top 5 Stocks Return", {}), "Top 5 Stocks (predicted % returns):"),
    ]
    if inference_meta:
        used = ", ".join(sorted(inference_meta.keys()))
        chunks.append(f"For inference we used offline lookback windows for tickers: {used}.")
    return chunks

def build_corpus():
    return base_chunks() + dynamic_chunks()

# =========================================================
#                  Embeddings & Index
# =========================================================
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def _embed(texts, embedder):
    embs = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return np.asarray(embs, dtype=np.float32)

def _build_index(embs: np.ndarray):
    if _HAS_FAISS:
        idx = faiss.IndexFlatIP(embs.shape[1])  # cosine since normalized
        idx.add(embs)
        return ("faiss", idx)
    return ("numpy", embs)

def _search(query_emb: np.ndarray, backend, store, k: int = 4):
    if backend == "faiss":
        D, I = store.search(query_emb.astype(np.float32), k)
        return I[0], D[0]
    sims = (store @ query_emb[0])  # cosine
    I = np.argsort(-sims)[:k]
    D = sims[I]
    return I, D

@st.cache_resource(show_spinner=False)
def get_index_and_corpus():
    corpus = build_corpus()
    embedder = get_embedder()
    embs = _embed(corpus, embedder)
    backend, store = _build_index(embs)
    return {"backend": backend, "store": store, "corpus": corpus, "embedder": embedder, "embs": embs}

def _corpus_signature():
    try:
        sig = {
            "weights": [crypto_weight, gold_weight, stock_weight],
            "top": {
                "gold":   gold_res.get("Top 5 Gold Return", {}),
                "crypto": crypto_res.get("Top 5 Crypto Return", {}),
                "stocks": stock_res.get("Top 5 Stocks Return", {}),
            },
            "meta_keys": sorted(list(inference_meta.keys())),
        }
        return json.dumps(sig, sort_keys=True)
    except Exception:
        return ""

if "corpus_sig" not in st.session_state:
    st.session_state.corpus_sig = ""

current_sig = _corpus_signature()
if st.session_state.corpus_sig != current_sig:
    get_index_and_corpus.clear()  # rebuild cache
    st.session_state.corpus_sig = current_sig

bundle = get_index_and_corpus()
CORPUS = bundle["corpus"]
EMBEDDER = bundle["embedder"]
BACKEND = bundle["backend"]
STORE = bundle["store"]

def retrieve(query, k=4):
    q_emb = _embed([query], EMBEDDER)
    I, D = _search(q_emb, BACKEND, STORE, k=k)
    ctx = [CORPUS[i] for i in I]
    return ctx, (I, D)

# =========================================================
#                  LLM (Groq) wrapper
# =========================================================
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # GPT-OSS via Groq

def _answer_from_context_only(full_prompt: str) -> str:
    # Extract and reuse context without calling an external LLM
    ctx = []
    qn = ""
    try:
        if "--- Context ---" in full_prompt:
            ctx_part = full_prompt.split("--- Context ---", 1)[1]
            if "--- Question ---" in ctx_part:
                ctx_text, qn = ctx_part.split("--- Question ---", 1)
            else:
                ctx_text, qn = ctx_part, ""
            ctx = [line.strip() for line in ctx_text.splitlines() if line.strip()]
    except Exception:
        pass

    bullets = "\n".join(f"- {c}" for c in ctx[:6])
    qn = qn.strip()
    pre = "GROQ_API_KEY not set — answering from local context only.\n\n"
    if qn:
        return pre + f"**Question:** {qn}\n\n**Key points from context:**\n{bullets}\n\n**Answer (context-based):**\n" \
               "Here’s a plain-English explanation based on the points above. " \
               "These are general concepts, not advice. Predictions are uncertain."
    return pre + "Using local context only. (Provide a question above and press *Get Answer*.)"

def groq_answer(prompt, model=DEFAULT_GROQ_MODEL, temperature=0.4, max_tokens=400):
    api_key = _get_groq_key()
    if not api_key:
        return _answer_from_context_only(prompt)
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You are a concise, neutral finance educator. Do not give investment advice. "
                 "Explain clearly with simple language and short paragraphs."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(Groq error: {e})"

# =========================================================
#             Auto summary (if session has data)
# =========================================================
if "rephrased_full_output" not in st.session_state:
    data_text = (
        _fmt_top5(gold_res.get("Top 5 Gold Return", {}),    "Top 5 Gold (predicted %):") + "\n\n" +
        _fmt_top5(crypto_res.get("Top 5 Crypto Return", {}),"Top 5 Crypto (predicted %):") + "\n\n" +
        _fmt_top5(stock_res.get("Top 5 Stocks Return", {}), "Top 5 Stocks (predicted %):")
    )
    have_any = any([
        gold_res.get("Top 5 Gold Return"),
        crypto_res.get("Top 5 Crypto Return"),
        stock_res.get("Top 5 Stocks Return"),
    ])
    if have_any:
        ctx, _ = retrieve("portfolio diversification, volatility, expected returns, moving averages", k=6)
        used_inputs = ", ".join(sorted(inference_meta.keys())) if inference_meta else "none"
        auto_prompt = f"""Summarize this performance snapshot in plain English for a beginner.
Be educational, not advisory. Note that predictions are uncertain.

--- Context ---
{chr(10).join(ctx)}

--- Weights ---
Crypto: {crypto_weight:.2f} | Gold: {gold_weight:.2f} | Stocks: {stock_weight:.2f}

--- Inputs used ---
Lookback windows available for: {used_inputs}

--- Data ---
{data_text}
"""
        with st.spinner("Summarizing portfolio results…"):
            st.session_state.rephrased_full_output = groq_answer(
                auto_prompt,
                model=DEFAULT_GROQ_MODEL,
                temperature=0.3,
                max_tokens=350
            )
    else:
        st.info("Run the quiz and LSTM forecasts first on the **main** page to see a summary here.")

if st.session_state.get("rephrased_full_output"):
    st.subheader("📊 Rephrased Asset Summary")
    st.write(st.session_state.rephrased_full_output)

# =========================================================
#                   Q&A Interface
# =========================================================
st.divider()
st.subheader("🧠 Ask anything (education only)")

with st.expander("Advanced settings", expanded=False):
    model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-120b"],
        index=0,
        help="Uses GPT-OSS via Groq."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens", 100, 1000, 450, 50)
    k_ctx = st.slider("Context passages (k)", 2, 10, 4, 1)

# Quick question presets
cols = st.columns(4)
with cols[0]:
    if st.button("What is a moving average?"):
        st.session_state["preset_q"] = "What does a moving average tell me and how is it used?"
with cols[1]:
    if st.button("What is diversification?"):
        st.session_state["preset_q"] = "Explain diversification and why it matters."
with cols[2]:
    if st.button("Why is crypto volatile?"):
        st.session_state["preset_q"] = "Why are cryptocurrencies so volatile?"
with cols[3]:
    if st.button("Gold in inflation?"):
        st.session_state["preset_q"] = "How can gold behave during inflation?"

user_q = st.text_area(
    "Your question",
    height=140,
    value=st.session_state.get("preset_q", ""),
    placeholder="e.g., What does a moving average tell me?"
)

if st.button("Get Answer"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        ctx, (I, D) = retrieve(user_q, k=k_ctx)
        prompt = f"""Answer as a finance educator (not an advisor). Use the context, cite concepts plainly,
and avoid recommendations. If uncertain, say so briefly.

--- Context ---
{chr(10).join(ctx)}

--- Question ---
{user_q}
"""
        with st.spinner("Thinking…"):
            ans = groq_answer(prompt, model=model_name, temperature=temperature, max_tokens=max_tokens)
        st.write(ans)

        with st.expander("Show retrieval context", expanded=False):
            st.caption("Top passages (highest similarity first):")
            for rank, i in enumerate(I, start=1):
                st.markdown(f"**{rank}.** {CORPUS[i]}")
