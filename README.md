<div align="center">

# ⚡ RankCraft-AI
### E-Commerce Search Query Re-Ranking & Click-Through (CTR) Predictor

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://rankcraft-ai-x9es.onrender.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.152-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.31-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](LICENSE)

*An industrial multi-stage search query re-ranking and click-through (CTR) prediction engine mirroring the core search architectures at Amazon and Google.*

### [👉 Click Here to Open Live Interactive Demo](https://rankcraft-ai-x9es.onrender.com)

[🚀 Live Demo](https://rankcraft-ai-x9es.onrender.com) • [Resume Impact & Architecture](RESUME_IMPACT.md) • [Evaluation Benchmarks](#-evaluation-benchmarks) • [Quickstart](#-local-quickstart)

</div>

---

## 📌 The Problem

When users search an e-commerce catalog (e.g., *"lightweight running shoes with arch support"*), keyword matching (BM25 / Elasticsearch) returns 100 loose matches. To maximize conversion, user satisfaction, and gross merchandise value (GMV), the platform must **re-order those candidates by predicting which product the user is most likely to click, add to cart, and purchase**.

---

## 🏛️ The 4 Pillars in Action

```
Raw Search Query ➔ [Pillar 2: NLP Intelligence] ➔ [Stage 1: BM25 Retrieval (Top 80)]
                                                         │
               ┌─────────────────────────────────────────┴─────────────────────────┐
               ▼                                                                   ▼
[Pillar 3: Deep Learning (MiniLM)]                               [Pillar 1: Tabular Feature Engineering]
- 384-dim Dense Vector Embeddings                               - Price elasticity to category mean/median
- Sub-5ms Cosine Semantic Similarity                            - Empirical Bayes smoothed historical CTR/CVR
- Fine-grained Intent-to-Title Match                            - Social proof: rating × log1p(reviews)
               │                                                                   │
               └─────────────────────────────────────────┬─────────────────────────┘
                                                         ▼
                                       [Pillar 4: Machine Learning (LTR)]
                                       - LambdaMART (LightGBM Ranker)
                                       - Pairwise Lambda Gradients (|ΔNDCG|)
                                       - Calibrated Predicted CTR % Probability
                                                         ▼
                                           Ranked E-Commerce Search Results
```

### 1. Data Science & Feature Engineering
* **Realistic Catalog & Sessions**: 2,060+ curated products across 6 major e-commerce categories (Footwear, Audio, Computing, Office Furniture, Apparel, Fitness).
* **Multi-Tier Relevance Ground Truth**: Graded relevance scale matching industry standards:
  * `0` = Impression without interaction (skipped)
  * `1` = Click (implicit interest)
  * `2` = Add-to-Cart (strong purchase intent)
  * `3` = Purchase (conversion signal)
* **17+ Domain Features**:
  * `price_diff_to_cat_mean`: $(p - \mu_{\text{cat}}) / \mu_{\text{cat}}$
  * `discount_pct`: percentage off original listing
  * `historical_ctr`: Empirical Bayes Laplace smoothed click affinity $(\text{clicks} + 5) / (\text{impr} + 150)$
  * `historical_cvr`: smoothed purchase conversion rate
  * `social_proof`: rating $\times \log(1 + \text{review\_count})$
  * `freshness_score`: exponential time-decay $\exp(-0.005 \times \text{days})$

### 2. NLP & Query Intelligence
* **Typo Correction**: Domain vocabulary distance matcher (e.g., `"nikke runing"` ➔ `"nike running"`, `"lighweight suport"` ➔ `"lightweight support"`).
* **Entity & Intent Extraction**:
  * **Brand Detection**: Nike, Sony, Apple, Bose, Logitech, Salomon, etc.
  * **Category Mapping**: Maps intent to catalog taxonomy.
  * **Attribute Parsing**: Extracts features like `lightweight`, `arch support`, `waterproof`, `wireless`.
  * **Price Constraints**: Regex parsing for `under $120`, `below 50`.
* **Lexical Matching**: BM25Okapi scoring and Jaccard token overlap.

### 3. Deep Learning (Semantic Match)
* **Model**: Sentence Transformer (`all-MiniLM-L6-v2`) generating 384-dimensional dense vectors.
* **Precomputed Caching**: Entire catalog pre-embedded and cached in `data/catalog_embeddings.npy` for sub-5ms dot-product matrix cosine scoring.
* **Semantic Interaction**: Captures conceptual relationships keyword search misses completely (e.g. associating "arch support" with "podiatrist-designed anatomical footbed").

### 4. Machine Learning: Learning-to-Rank (LTR)
* **Algorithm**: LambdaMART (`lightgbm.LGBMRanker`) with `objective="lambdarank"` and `metric="ndcg"`.
* **Pairwise $\lambda$-Gradients**: Directly penalizes rank inversions weighted by the change in NDCG ($|\Delta\text{NDCG}|$).
* **Evaluation**: Evaluated on holdout queries using **NDCG@5, NDCG@10, MRR, and MAP@10**.

---

## 📊 Evaluation Benchmarks

Rigorous benchmark evaluation across holdout test query sessions:

| Retrieval Method | NDCG@5 | NDCG@10 | MRR | MAP@10 | Latency (p99) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Keyword Baseline** | 0.4535 | 0.4836 | 0.9000 | 0.7432 | ~4.2 ms |
| **Dense Semantic Vector Search (MiniLM)** | 0.5336 | 0.5187 | 0.8519 | 0.7349 | ~6.8 ms |
| **RankCraft LambdaMART LTR (Multi-Modal)** | **0.9095** | **0.8596** | **0.9697** | **0.9052** | **~9.2 ms** |
| **Relative Lift over BM25 Baseline** | **+100.5%** | **+77.7%** | **+7.7%** | **+21.8%** | *Ultra-Fast* |

### Global Feature Importance (LightGBM Gain):
1. **Discount Percentage** (25.9%)
2. **BM25 Lexical Score** (21.2%)
3. **Jaccard Token Overlap** (9.7%)
4. **Price Difference to Category Mean** (6.4%)
5. **Historical CTR** (6.1%)
6. **Semantic Similarity Cosine** (6.1%)
7. **Historical CVR** (4.9%)
8. **Product Freshness Decay** (4.3%)

---

## ✨ Luxury Frontend Features

* **Tri-View & Side-by-Side Comparison**: Switch between LambdaMART, Pure Semantic, BM25, or compare them directly side-by-side.
* **Live Rank Deltas**: Badges displaying position changes (`▲ +33 spots` or `▼ -12 spots`).
* **Predicted CTR Gauges**: Calibrated probability meters showing likelihood of user click.
* **NLP Query Inspector**: Real-time display of typo fixes, detected brand, category, attributes, and price bounds.
* **Slide-over ML Feature Attribution Drawer**: Inspect exact feature values (semantic score, price delta, historical CTR) for any product.
* **Interactive What-If CTR Simulator**: Sliders to adjust price, discount, rating, and reviews in real time to see the simulated CTR shift!
* **Resume & System Design Modal**: 1-click copyable resume bullet points and ML interview questions.

---

## 🚀 Local Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/milind/RankCraft-AI.git
cd RankCraft-AI

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
python run.py --port 7860
```
Open **`http://localhost:7860`** in your browser!

---

## 🌐 Live Production Deployment
The live application is deployed and running 24/7 on Render:

🔗 **[https://rankcraft-ai-x9es.onrender.com](https://rankcraft-ai-x9es.onrender.com)**

---

## 🌐 Cloud Deployment Options

### Option A: Hugging Face Spaces (Recommended - 100% Free)
Hugging Face Spaces provides **16 GB RAM + 2 vCPU** for free with persistent URLs:

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New Space** ➔ Name it `rankcraft-ecommerce-reranker`.
3. Select **Docker** as the Space SDK and choose **Blank** template.
4. Clone your new Hugging Face Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/rankcraft-ecommerce-reranker
   cd rankcraft-ecommerce-reranker
   ```
5. Copy all files from this repository into your Space folder:
   ```bash
   cp -r /path/to/RankCraft-AI/* .
   ```
6. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy RankCraft-AI Search Re-Ranker"
   git push origin main
   ```
7. Hugging Face builds the Docker container and launches your app at:
   `https://huggingface.co/spaces/<your-username>/rankcraft-ecommerce-reranker`

### Option B: Render (Free Web Service)
1. Push this repository to GitHub.
2. Sign in to [render.com](https://render.com/) and click **New ➔ Web Service**.
3. Select your GitHub repository.
4. Set Environment to **Docker** and Port to `7860`.
5. Click **Create Web Service** — Render automatically builds and hosts your app for free!

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/search` | `POST` | Re-ranks catalog for a query with NLP analysis & tri-view results |
| `/api/simulate` | `POST` | What-If sandbox recalculating CTR for price/rating adjustments |
| `/api/metrics` | `GET` | Returns NDCG@5, NDCG@10, MRR, MAP benchmarks & lifts |
| `/api/features/importance` | `GET` | Returns LightGBM feature gain and split count attributions |
| `/api/health` | `GET` | Service readiness, device, and catalog status |

### Example Search Request:
```bash
curl -X POST http://localhost:7860/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "lightweight running shoes with arch support", "top_k": 80, "limit": 10}'
```

---

## 📄 License
MIT License. Created by Milind for modern search and recommendation systems portfolio.
