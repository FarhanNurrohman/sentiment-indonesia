# 🇮🇩 Indonesian Sentiment Analysis API

Multi-model sentiment analysis for Indonesian text, deployed as **two independent services**:

| Service | Models | Platform | Live Demo |
|---|---|---|---|
| **Web UI** | SVM, Naive Bayes, IndoBERT | HF Spaces (Gradio + ZeroGPU) | [sentiment-indo.hf.space](https://huggingface.co/spaces/USERNAME/sentiment-indo) |
| **REST API** | SVM, Naive Bayes | Render (Docker) | [sentiment-svm-nb.onrender.com](https://sentiment-svm-nb.onrender.com) |

---

## 🎯 What This Project Demonstrates

This is not a notebook-turned-API. It's a **production-oriented ML serving stack** built to answer one question: *how do you take a trained model and make it reliably usable by others?*

**Engineering decisions made along the way:**

- **Training-serving consistency** — verified empirically that notebook predictions match endpoint predictions on held-out test sentences. No silent skew.
- **Failure isolation** — IndoBERT and classical models run on **separate deployments**. If IndoBERT fails to load, SVM and NB stay up.
- **Trimmed dependencies** — `requirements.txt` reduced from 44 lines (full training env) to 13 lines (runtime only). Docker image dropped from ~2 GB to ~300 MB.
- **Observability from day one** — structured logging with request-ID tracing, `LOG_LEVEL` env var, `/health` readiness probe.
- **Environment-based config** — zero hardcoded ports, paths, or secrets. Same code runs locally and on any platform.

---

## 🏗️ Architecture
┌─────────────────────────────────────┐
│ User / Client │
└──────────────┬──────────────────────┘
│
┌────────────────────┼────────────────────┐
│ │
▼ ▼
┌──────────────────────┐ ┌──────────────────────┐
│ HF Spaces (Gradio) │ │ Render (Docker) │
│ ZeroGPU │ │ Free tier 512 MB │
│ │ │ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ SVM │ │ │ │ SVM │ │
│ ├────────────────┤ │ │ ├────────────────┤ │
│ │ Naive Bayes │ │ │ │ Naive Bayes │ │
│ ├────────────────┤ │ │ └────────────────┘ │
│ │ IndoBERT │ │ │ │
│ │ (NumPy, 440MB) │ │ │ Flask + Waitress │
│ └────────────────┘ │ │ REST API │
│ │ │ │
│ Gradio UI │ │ /health │
│ (dropdown + input) │ │ /predict/svm │
└──────────────────────┘ │ /predict/naive-bayes│
└──────────────────────┘


**Why two services?**
- IndoBERT needs ~1 GB RAM — exceeds Render free tier's 512 MB.
- HF Spaces ZeroGPU handles the memory-heavy model; Render handles the lightweight REST endpoints.
- Separate deployments = **independent failure domains**. One service can be upgraded or restarted without affecting the other.

---

## 🤖 Models

### 1. Multi-Class SVM (from scratch)
- **Implementation**: custom `MultiClassSVM_Vectorized` — no `sklearn.svm`.
- **Features**: TF-IDF (fit on training set only, saved as part of the bundle).
- **Task**: binary sentiment (negative vs non-negative).

### 2. Complement Naive Bayes (from scratch)
- **Implementation**: custom `ComplementNaiveBayes` — no `sklearn.naive_bayes`.
- **Suitable for**: imbalanced datasets (complement NB weights the *complement* of each class).
- **Features**: same TF-IDF bundle as SVM.

### 3. IndoBERT (from scratch, NumPy serving)
- **Implementation**: full BERT architecture written from scratch — embedding, 12 encoder layers, multi-head attention, pooler, classifier head. No HuggingFace `AutoModel`.
- **Training**: CuPy (GPU) on Kaggle.
- **Serving**: NumPy (CPU) — converted for portability, weights verified against training output.
- **Weights**: 440 MB, loaded once at startup.

**Why from scratch?** To verify understanding of transformer internals — attention math, layer norm, GELU, positional encoding. The skill is not just "call `AutoModel.from_pretrained`" but "know what's inside it."

---

## 🚀 REST API (Render)

**Base URL**: `https://sentiment-svm-nb.onrender.com`

> **Note**: free tier sleeps after 15 minutes of inactivity. First request after idle takes ~30 seconds (cold start).

### `GET /health`

```bash
curl https://sentiment-svm-nb.onrender.com/health