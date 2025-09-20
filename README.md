# Wealth Genie 💹

Wealth Genie is an **educational portfolio analysis and forecasting app** built with [Streamlit](https://streamlit.io/).  
It combines **risk profiling, predictive modeling, interactive graphs, and an AI-powered chatbot** to help users learn about portfolio allocation and market behavior.

⚠️ **Disclaimer:** This project is for **educational purposes only** and is **not financial advice**.

---

## ✨ Features

- **Risk Profiling:**  
  13-question quiz to calculate a normalized risk score between 0.0 (conservative) and 1.0 (aggressive).

- **Asset Allocation:**  
  Risk score determines portfolio weights across **Gold, Stocks, and Crypto**.

- **Forecasting Engine:**  
  Uses **LSTM deep learning models** with historical CSV lookbacks and scalers to predict short-term returns.

- **Visual Insights:**  
  Interactive **Altair graphs** show portfolio allocation and top asset predictions.

- **Educational Chatbot:**  
  Retrieval-augmented chatbot powered by **SentenceTransformers embeddings** + **Groq API**.  
  Provides plain-English explanations of investment concepts (no financial advice).

---

## 📂 Project Structure

```
New_Project/
├── main.py                   # Main quiz + forecasting logic
├── pretrained_loader.py       # Model loading helper
├── pages/                     # Additional Streamlit pages (graphs, chatbot, etc.)
├── artifacts/                 # Pre-trained models
├── lookback/                  # Historical CSVs for inference
├── price_cache/               # Cached price data
├── Flowchart.pdf              # High-level architecture diagram
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## 🚀 Getting Started

### 1. Clone repository
```bash
git clone https://github.com/yourusername/wealth-genie.git
cd wealth-genie
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data
- Place lookback CSVs under `lookback/`.
- Place pre-trained models under `artifacts-{ticker}/` or `artifacts/{ticker}/`.

### 4. Run the app
Start Streamlit:
```bash
streamlit run main.py
```

Then open:
- **Main page (Quiz + Forecasts)** → `main.py`
- **Graphs page** → inside `pages/`
- **Chatbot page** → inside `pages/`

---

## 🔑 API Keys

- **Groq API:**  
  Set via environment variable or `st.secrets`:
  ```bash
  export GROQ_API_KEY="your_api_key"
  ```
  or inside `.streamlit/secrets.toml`:
  ```toml
  GROQ_API_KEY="your_api_key"
  ```

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) – Web app framework  
- [TensorFlow / Keras](https://www.tensorflow.org/) – LSTM forecasting models  
- [Joblib](https://joblib.readthedocs.io/) – Model & scaler persistence  
- [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) – Data handling  
- [Altair](https://altair-viz.github.io/) – Interactive charts  
- [SentenceTransformers](https://www.sbert.net/) – Embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) – Fast similarity search (optional, falls back to NumPy)  
- [Groq API](https://groq.com/) – LLM backend for chatbot  
- [yfinance](https://pypi.org/project/yfinance/) – Market data fetching  
- [scikit-learn](https://scikit-learn.org/) – Feature scaling, ML utilities  

---

## ⚠️ Disclaimer

This app is **not** financial advice. Predictions are uncertain,  
and outputs should be considered **educational only**.
# wealth-genie
# wealth-genie
# wealth-genie
# wealth-genie
# wealth-genie1
# wealth-genie1
# wealth-genie1
