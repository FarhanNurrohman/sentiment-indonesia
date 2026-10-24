# 🇮🇩 Indonesian Sentiment Analysis API

Multi-model sentiment analysis for Indonesian text, deployed as **two independent services**:

| Service | Models | Platform | Live Demo |
|---|---|---|---|
| **Web UI** | SVM, Naive Bayes, IndoBERT | HF Spaces (Gradio + ZeroGPU) | [Sentiment App](https://farhannurrahman95-sentiment-resetindonesia.hf.space) |
| **REST API** | SVM, Naive Bayes | SnapDeploy (Docker) | [API Endpoint](https://sentiment-indonesia-8b9d3.containers.snapdeploy.app) |

*Repository for the Web UI: [Hugging Face Space](https://huggingface.co/spaces/farhannurrahman95/sentiment-resetindonesia)*

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

```text
                    ┌─────────────────────────────────────┐
                    │         User / Client               │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                         │
              ▼                                         ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │  HF Spaces (Gradio)  │              │  SnapDeploy (Docker) │
   │  ZeroGPU             │              │  Free tier 512 MB    │
   │                      │              │                      │
   │  ┌────────────────┐  │              │  ┌────────────────┐  │
   │  │ SVM            │  │              │  │ SVM            │  │
   │  ├────────────────┤  │              │  ├────────────────┤  │
   │  │ Naive Bayes    │  │              │  │ Naive Bayes    │  │
   │  ├────────────────┤  │              │  └────────────────┘  │
   │  │ IndoBERT       │  │              │                      │
   │  │ (NumPy, 440MB) │  │              │  Flask + Waitress    │
   │  └────────────────┘  │              │  REST API            │
   │                      │              │                      │
   │  Gradio UI           │              │  /health             │
   │  (dropdown + input)  │              │  /predict/svm        │
   └──────────────────────┘              │  /predict/naive-bayes│
                                         └──────────────────────┘
