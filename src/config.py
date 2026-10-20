"""
Configuration and constants for RankCraft-AI
E-Commerce Search Query Re-Ranking & Click-Through (CTR) Predictor
"""
import os
# Configure OpenMP and thread settings to ensure thread safety across PyTorch and LightGBM on macOS & Linux
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

CATALOG_PATH = DATA_DIR / "catalog.json"
CATALOG_EMBEDDINGS_PATH = DATA_DIR / "catalog_embeddings.npy"
SEARCH_LOGS_PATH = DATA_DIR / "search_logs.json"
LTR_MODEL_PATH = MODELS_DIR / "ltr_lambdamart.joblib"
LTR_BOOSTER_PATH = MODELS_DIR / "ltr_lambdamart.txt"
METRICS_PATH = MODELS_DIR / "evaluation_results.json"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.json"

# Deep Learning / NLP Models
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Candidate Retrieval
DEFAULT_TOP_K_CANDIDATES = 100
DEFAULT_DISPLAY_LIMIT = 24

# LTR Features list
FEATURE_NAMES = [
    "semantic_similarity",       # Pillar 3: Deep Sentence Transformer Cosine Score
    "bm25_score",                 # Pillar 1/2: Lexical BM25 score
    "jaccard_overlap",            # Pillar 2: Token overlap between query & title
    "brand_match",                # Pillar 2: Exact brand detection match
    "category_match",             # Pillar 2: Intent category match
    "exact_query_in_title",       # Pillar 2: Substring query match in title
    "price_normalized",           # Pillar 1: Min-Max normalized price
    "price_diff_to_cat_mean",     # Pillar 1: (price - cat_avg) / cat_avg
    "price_diff_to_cat_median",   # Pillar 1: (price - cat_med) / cat_med
    "discount_pct",               # Pillar 1: Percentage off original price
    "rating",                     # Pillar 1: Product star rating (1.0 to 5.0)
    "log_review_count",           # Pillar 1: log1p(review_count)
    "historical_ctr",             # Pillar 1: Smoothed empirical Bayes click rate
    "historical_cvr",             # Pillar 1: Conversion rate (purchases/clicks)
    "is_prime_eligible",          # Pillar 1: Fast shipping indicator
    "freshness_score",            # Pillar 1: Time decay score
    "return_rate"                 # Pillar 1: Product return rate penalty
]

# Relevance Grading Scale (Industrial E-Commerce standard: Amazon/Google)
# 0 = Impression without interaction (irrelevant or skipped)
# 1 = Click (implicit interest)
# 2 = Add to Cart (high purchase intent)
# 3 = Purchase (verified relevance)
RELEVANCE_LABELS = {
    "impression": 0,
    "click": 1,
    "cart": 2,
    "purchase": 3
}

# LambdaMART Hyperparameters
LAMBDAMART_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": 0.05,
    "n_estimators": 140,
    "num_leaves": 31,
    "min_child_samples": 12,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "verbose": -1
}
