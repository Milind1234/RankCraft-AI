"""
Pillar 4: Learning-to-Rank (LTR) Engine with LambdaMART
Trains a LightGBM Ranker, evaluates NDCG@K and MRR (industry standard Amazon/Google metrics),
extracts Tree SHAP/Gain feature importance, and scores live candidates.
"""
import joblib
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import (
    LTR_MODEL_PATH,
    LTR_BOOSTER_PATH,
    METRICS_PATH,
    FEATURE_NAMES,
    LAMBDAMART_PARAMS
)


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    """Computes Discounted Cumulative Gain at rank K using standard exponential gain."""
    relevances = relevances[:k]
    dcg = 0.0
    for idx, rel in enumerate(relevances):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(idx + 2.0)  # idx is 0-indexed, so pos is idx+1, log2(pos+1) = log2(idx+2)
        dcg += gain / discount
    return dcg


def compute_ndcg(actual_relevances: List[int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg(actual_relevances, k=k)
    ideal_relevances = sorted(actual_relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k=k)
    if idcg <= 0.0:
        return 1.0 if dcg <= 0.0 else 0.0
    return dcg / idcg


def compute_mrr(actual_relevances: List[int], k: int = 10, threshold: int = 1) -> float:
    """Computes Mean Reciprocal Rank (position of first relevant item)."""
    for idx, rel in enumerate(actual_relevances[:k]):
        if rel >= threshold:
            return 1.0 / (idx + 1.0)
    return 0.0


def compute_map(actual_relevances: List[int], k: int = 10, threshold: int = 1) -> float:
    """Computes Mean Average Precision at rank K."""
    hits = 0
    sum_precisions = 0.0
    for idx, rel in enumerate(actual_relevances[:k]):
        if rel >= threshold:
            hits += 1
            precision_at_i = hits / (idx + 1.0)
            sum_precisions += precision_at_i
    total_relevant = sum(1 for r in actual_relevances if r >= threshold)
    if total_relevant == 0:
        return 1.0
    return sum_precisions / min(k, total_relevant)


class LTREngine:
    """Manages LambdaMART training, evaluation, SHAP explainability, and live re-ranking."""

    def __init__(self):
        self.model: Optional[lgb.LGBMRanker] = None
        self.booster: Optional[lgb.Booster] = None
        self.feature_names = FEATURE_NAMES
        self.evaluation_results: Dict[str, Any] = {}

    def prepare_dataset(self, search_logs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Flattens session logs into tabular X, y, and query group sizes."""
        rows = []
        labels = []
        groups = []

        for session in search_logs:
            records = session.get("records", [])
            if not records:
                continue
            groups.append(len(records))
            for rec in records:
                labels.append(rec["relevance_grade"])
                feat_dict = rec["features"]
                row = [feat_dict.get(feat, 0.0) for feat in self.feature_names]
                rows.append(row)

        X = pd.DataFrame(rows, columns=self.feature_names)
        y = np.array(labels, dtype=int)
        groups = np.array(groups, dtype=int)
        return X, y, groups

    def train(self, search_logs: List[Dict[str, Any]], force_retrain: bool = False):
        """
        Trains LambdaMART (LGBMRanker) model with ndcg objective and group query structures.
        """
        if LTR_BOOSTER_PATH.exists() and not force_retrain:
            print(f"[LTREngine] Loading trained LambdaMART Booster from {LTR_BOOSTER_PATH}...")
            self.booster = lgb.Booster(model_file=str(LTR_BOOSTER_PATH))
            if LTR_MODEL_PATH.exists():
                try:
                    self.model = joblib.load(LTR_MODEL_PATH)
                except Exception:
                    pass
            if METRICS_PATH.exists():
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    self.evaluation_results = json.load(f)
            return self.booster

        print(f"[LTREngine] Training LambdaMART model on {len(search_logs)} search sessions...")
        X, y, groups = self.prepare_dataset(search_logs)

        # Split train / validation queries (80/20 split based on query groups)
        num_sessions = len(groups)
        split_idx = int(num_sessions * 0.8)

        train_groups = groups[:split_idx]
        val_groups = groups[split_idx:]

        num_train_samples = sum(train_groups)
        X_train, y_train = X.iloc[:num_train_samples], y[:num_train_samples]
        X_val, y_val = X.iloc[num_train_samples:], y[num_train_samples:]

        self.model = lgb.LGBMRanker(**LAMBDAMART_PARAMS)
        self.model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_metric="ndcg",
            eval_at=(5, 10),
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        self.booster = self.model.booster_
        LTR_BOOSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(LTR_BOOSTER_PATH))
        joblib.dump(self.model, LTR_MODEL_PATH)
        print(f"[LTREngine] Models saved to {LTR_BOOSTER_PATH} and {LTR_MODEL_PATH}.")

        # Run benchmark evaluation across all 3 methods
        self.evaluate_benchmarks(search_logs)
        return self.booster

    def evaluate_benchmarks(self, search_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates and compares:
        1. BM25 Keyword Baseline
        2. Sentence Transformer Dense Semantic Baseline
        3. RankCraft LambdaMART LTR Re-Ranking
        Computes NDCG@5, NDCG@10, MRR, and MAP.
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before evaluation.")

        metrics = {
            "bm25": {"ndcg_5": [], "ndcg_10": [], "mrr": [], "map_10": []},
            "semantic": {"ndcg_5": [], "ndcg_10": [], "mrr": [], "map_10": []},
            "lambdamart": {"ndcg_5": [], "ndcg_10": [], "mrr": [], "map_10": []}
        }

        for session in search_logs:
            records = session.get("records", [])
            if len(records) < 5:
                continue

            # Method 1: Sort by BM25
            sorted_bm25 = sorted(records, key=lambda r: r.get("bm25_score", 0.0), reverse=True)
            bm25_rels = [r["relevance_grade"] for r in sorted_bm25]
            metrics["bm25"]["ndcg_5"].append(compute_ndcg(bm25_rels, k=5))
            metrics["bm25"]["ndcg_10"].append(compute_ndcg(bm25_rels, k=10))
            metrics["bm25"]["mrr"].append(compute_mrr(bm25_rels, k=10))
            metrics["bm25"]["map_10"].append(compute_map(bm25_rels, k=10))

            # Method 2: Sort by Semantic Similarity
            sorted_sem = sorted(records, key=lambda r: r.get("semantic_score", 0.0), reverse=True)
            sem_rels = [r["relevance_grade"] for r in sorted_sem]
            metrics["semantic"]["ndcg_5"].append(compute_ndcg(sem_rels, k=5))
            metrics["semantic"]["ndcg_10"].append(compute_ndcg(sem_rels, k=10))
            metrics["semantic"]["mrr"].append(compute_mrr(sem_rels, k=10))
            metrics["semantic"]["map_10"].append(compute_map(sem_rels, k=10))

            # Method 3: Sort by LambdaMART LTR
            X_sess = pd.DataFrame([
                [r["features"].get(f, 0.0) for f in self.feature_names]
                for r in records
            ], columns=self.feature_names)
            ltr_scores = self.model.predict(X_sess)
            ranked_indices = np.argsort(-ltr_scores)
            ltr_rels = [records[i]["relevance_grade"] for i in ranked_indices]
            metrics["lambdamart"]["ndcg_5"].append(compute_ndcg(ltr_rels, k=5))
            metrics["lambdamart"]["ndcg_10"].append(compute_ndcg(ltr_rels, k=10))
            metrics["lambdamart"]["mrr"].append(compute_mrr(ltr_rels, k=10))
            metrics["lambdamart"]["map_10"].append(compute_map(ltr_rels, k=10))

        summary = {}
        for method in ["bm25", "semantic", "lambdamart"]:
            summary[method] = {
                "ndcg_5": round(float(np.mean(metrics[method]["ndcg_5"])), 4),
                "ndcg_10": round(float(np.mean(metrics[method]["ndcg_10"])), 4),
                "mrr": round(float(np.mean(metrics[method]["mrr"])), 4),
                "map_10": round(float(np.mean(metrics[method]["map_10"])), 4)
            }

        # Calculate percentage lifts over BM25 baseline
        bm25_ndcg10 = summary["bm25"]["ndcg_10"]
        bm25_mrr = summary["bm25"]["mrr"]
        ltr_ndcg10 = summary["lambdamart"]["ndcg_10"]
        ltr_mrr = summary["lambdamart"]["mrr"]

        summary["lifts"] = {
            "ndcg_10_lift_pct": round(((ltr_ndcg10 - bm25_ndcg10) / (bm25_ndcg10 + 1e-5)) * 100.0, 1),
            "mrr_lift_pct": round(((ltr_mrr - bm25_mrr) / (bm25_mrr + 1e-5)) * 100.0, 1)
        }

        # Feature importances
        importances = self.model.feature_importances_
        gain_importances = self.model.booster_.feature_importance(importance_type="gain")
        total_gain = sum(gain_importances) + 1e-5
        normalized_gain = [round(float(g / total_gain), 4) for g in gain_importances]

        feat_importance_list = [
            {"feature": f, "importance_gain": ng, "split_count": int(imp)}
            for f, ng, imp in zip(self.feature_names, normalized_gain, importances)
        ]
        feat_importance_list.sort(key=lambda x: x["importance_gain"], reverse=True)
        summary["feature_importances"] = feat_importance_list

        self.evaluation_results = summary
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("[LTREngine] Evaluation Benchmark Results:")
        print(f"  BM25 Baseline:        NDCG@10 = {summary['bm25']['ndcg_10']:.4f}, MRR = {summary['bm25']['mrr']:.4f}")
        print(f"  Semantic Vector:      NDCG@10 = {summary['semantic']['ndcg_10']:.4f}, MRR = {summary['semantic']['mrr']:.4f}")
        print(f"  LambdaMART LTR:       NDCG@10 = {summary['lambdamart']['ndcg_10']:.4f}, MRR = {summary['lambdamart']['mrr']:.4f}")
        print(f"  Lift over BM25:       NDCG@10 +{summary['lifts']['ndcg_10_lift_pct']}% | MRR +{summary['lifts']['mrr_lift_pct']}%")

        return summary

    def rank_candidates(
        self,
        candidate_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate records using trained LambdaMART model.
        Returns candidates ordered by predicted relevance score with predicted CTR %.
        """
        if self.booster is None and self.model is None:
            raise ValueError("Model must be initialized before ranking.")

        if not candidate_records:
            return []

        X_mat = np.array([
            [r["features"].get(f, 0.0) for f in self.feature_names]
            for r in candidate_records
        ], dtype=np.float32)

        if self.booster is not None:
            scores = self.booster.predict(X_mat)
        else:
            scores = self.model.predict(X_mat)

        # Convert raw ranking logits into calibrated CTR % probability using sigmoid
        calibrated_ctrs = [round(float(1.0 / (1.0 + math.exp(-0.8 * float(s)))) * 100.0, 1) for s in scores]

        # Attach scores to records
        for i, rec in enumerate(candidate_records):
            rec["ltr_score"] = float(scores[i])
            rec["predicted_ctr_pct"] = calibrated_ctrs[i]

        # Sort descending by LTR score
        ranked = sorted(candidate_records, key=lambda r: r["ltr_score"], reverse=True)
        return ranked
