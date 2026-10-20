"""
Multi-Stage E-Commerce Search & Re-Ranking Pipeline Orchestrator
Coordinates:
  Stage 1: NLP Query Intelligence & Typo Correction
  Stage 2: BM25 Lexical Candidate Retrieval (Top 50-100)
  Stage 3: Deep Neural Semantic Scoring (all-MiniLM-L6-v2)
  Stage 4: LambdaMART LTR Re-Ranking & CTR Prediction
"""
import time
from typing import List, Dict, Any, Optional
from src.data_engine import DataEngine
from src.nlp_engine import NLPEngine, BM25Retriever
from src.deep_learning_engine import DeepLearningEngine
from src.ltr_engine import LTREngine
from src.config import (
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_DISPLAY_LIMIT,
    LTR_BOOSTER_PATH
)


class SearchPipeline:
    """End-to-end multi-stage search query re-ranking orchestrator."""

    def __init__(self):
        print("[SearchPipeline] Initializing RankCraft-AI Pipeline Components...")
        self.data_engine = DataEngine()
        self.catalog = self.data_engine.generate_or_load_catalog()

        self.nlp_engine = NLPEngine()
        self.bm25_retriever = BM25Retriever(self.catalog)

        self.deep_engine = DeepLearningEngine()
        self.deep_engine.build_or_load_catalog_embeddings(self.catalog)

        self.ltr_engine = LTREngine()

        # Load pre-trained booster directly to conserve memory on free cloud tiers
        if LTR_BOOSTER_PATH.exists():
            self.ltr_engine.train(None)
        else:
            search_logs = self.data_engine.generate_search_logs(
                nlp_cleaner=self.nlp_engine,
                deep_engine=self.deep_engine,
                bm25_retriever=self.bm25_retriever
            )
            self.ltr_engine.train(search_logs)

        import gc
        gc.collect()
        print("[SearchPipeline] Initialization complete! Pipeline ready for real-time inference.")

    def search_and_rerank(
        self,
        query: str,
        top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
        display_limit: int = DEFAULT_DISPLAY_LIMIT
    ) -> Dict[str, Any]:
        """
        Executes end-to-end search and re-ranking:
        1. NLP cleaning & intent extraction
        2. Stage 1: BM25 candidate retrieval (Top 50-100)
        3. Stage 2: Deep Semantic scoring
        4. Stage 3: Tabular feature extraction
        5. Stage 4: LambdaMART LTR Re-ranking with predicted CTR
        Returns comparative views for BM25, Pure Semantic, and LambdaMART.
        """
        start_time = time.time()

        # Step 1: NLP Query Intelligence
        cleaned_query, intent = self.nlp_engine.clean_and_extract_intent(query)

        # Step 2: Lexical Candidate Retrieval
        bm25_raw_candidates = self.bm25_retriever.retrieve(cleaned_query, top_k=top_k_candidates)

        if not bm25_raw_candidates:
            # Fallback to random sample if query has zero BM25 token matches
            bm25_raw_candidates = [dict(p) for p in self.catalog[:top_k_candidates]]
            for p in bm25_raw_candidates:
                p["bm25_score"] = 0.0

        # Step 3: Deep Semantic Scoring
        semantic_scores = self.deep_engine.compute_similarity(cleaned_query, bm25_raw_candidates)

        # Step 4: Tabular Feature Engineering & Candidate Record Assembly
        candidate_records = []
        for prod, sem_score in zip(bm25_raw_candidates, semantic_scores):
            bm25_score = prod.get("bm25_score", 0.0)
            feat_dict = self.data_engine.extract_tabular_features(prod, intent, sem_score, bm25_score)
            candidate_records.append({
                "product": prod,
                "product_id": prod["id"],
                "features": feat_dict,
                "semantic_score": round(float(sem_score), 4),
                "bm25_score": round(float(bm25_score), 4)
            })

        # Generate comparative Views:

        # View A: Pure BM25 Ranking
        sorted_by_bm25 = sorted(candidate_records, key=lambda r: r["bm25_score"], reverse=True)
        bm25_rank_map = {r["product_id"]: idx + 1 for idx, r in enumerate(sorted_by_bm25)}

        # View B: Pure Semantic Vector Ranking
        sorted_by_sem = sorted(candidate_records, key=lambda r: r["semantic_score"], reverse=True)
        sem_rank_map = {r["product_id"]: idx + 1 for idx, r in enumerate(sorted_by_sem)}

        # View C: LambdaMART Re-Ranking
        ranked_by_ltr = self.ltr_engine.rank_candidates(candidate_records)

        # Decorate each item with rank deltas and explainability metadata
        formatted_ltr = []
        for idx, rec in enumerate(ranked_by_ltr[:display_limit]):
            prod_id = rec["product_id"]
            prod = rec["product"]
            bm25_pos = bm25_rank_map.get(prod_id, 999)
            sem_pos = sem_rank_map.get(prod_id, 999)
            ltr_pos = idx + 1

            # Rank delta: positive means promoted higher up (e.g. from pos 10 to pos 2 is +8)
            rank_delta_bm25 = bm25_pos - ltr_pos

            formatted_ltr.append({
                "rank": ltr_pos,
                "bm25_rank": bm25_pos,
                "semantic_rank": sem_pos,
                "rank_delta": rank_delta_bm25,
                "predicted_ctr_pct": rec.get("predicted_ctr_pct", 50.0),
                "ltr_score": round(rec.get("ltr_score", 0.0), 3),
                "semantic_score": rec["semantic_score"],
                "bm25_score": rec["bm25_score"],
                "features": rec["features"],
                "product": prod
            })

        formatted_bm25 = []
        for idx, rec in enumerate(sorted_by_bm25[:display_limit]):
            formatted_bm25.append({
                "rank": idx + 1,
                "bm25_score": rec["bm25_score"],
                "semantic_score": rec["semantic_score"],
                "product": rec["product"]
            })

        formatted_sem = []
        for idx, rec in enumerate(sorted_by_sem[:display_limit]):
            formatted_sem.append({
                "rank": idx + 1,
                "semantic_score": rec["semantic_score"],
                "bm25_score": rec["bm25_score"],
                "product": rec["product"]
            })

        latency_ms = round((time.time() - start_time) * 1000.0, 1)

        return {
            "query": query,
            "query_analysis": intent,
            "latency_ms": latency_ms,
            "candidate_count": len(candidate_records),
            "catalog_total": len(self.catalog),
            "results": {
                "lambdamart": formatted_ltr,
                "semantic": formatted_sem,
                "bm25": formatted_bm25
            }
        }

    def simulate_what_if(
        self,
        product_id: str,
        query: str,
        price_override: Optional[float] = None,
        discount_override: Optional[float] = None,
        rating_override: Optional[float] = None,
        review_count_override: Optional[int] = None,
        prime_override: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Interactive CTR Sandbox:
        Recalculates LambdaMART score, predicted CTR %, and feature contributions
        when a user/recruiter tweaks product pricing, discount, rating, or reviews.
        """
        prod = self.data_engine.catalog_by_id.get(product_id)
        if not prod:
            return {"error": f"Product {product_id} not found"}

        # Clone product with overrides
        sim_prod = dict(prod)
        if price_override is not None:
            sim_prod["price"] = float(price_override)
        if discount_override is not None:
            sim_prod["discount_pct"] = float(discount_override)
        if rating_override is not None:
            sim_prod["rating"] = float(rating_override)
        if review_count_override is not None:
            sim_prod["review_count"] = int(review_count_override)
        if prime_override is not None:
            sim_prod["is_prime_eligible"] = bool(prime_override)

        cleaned_q, intent = self.nlp_engine.clean_and_extract_intent(query)
        sem_scores = self.deep_engine.compute_similarity(cleaned_q, [sim_prod])
        sem_score = sem_scores[0]
        bm25_candidates = self.bm25_retriever.retrieve(cleaned_q, top_k=80)
        bm25_score = next((c["bm25_score"] for c in bm25_candidates if c["id"] == product_id), 1.5)

        # Baseline original features
        orig_features = self.data_engine.extract_tabular_features(prod, intent, sem_score, bm25_score)
        orig_ranked = self.ltr_engine.rank_candidates([{
            "product_id": product_id,
            "features": orig_features,
            "semantic_score": sem_score,
            "bm25_score": bm25_score
        }])[0]

        # Simulated features
        sim_features = self.data_engine.extract_tabular_features(sim_prod, intent, sem_score, bm25_score)
        sim_ranked = self.ltr_engine.rank_candidates([{
            "product_id": product_id,
            "features": sim_features,
            "semantic_score": sem_score,
            "bm25_score": bm25_score
        }])[0]

        return {
            "product_id": product_id,
            "title": prod["title"],
            "query": query,
            "original": {
                "price": prod["price"],
                "discount_pct": prod["discount_pct"],
                "rating": prod["rating"],
                "review_count": prod["review_count"],
                "is_prime_eligible": prod["is_prime_eligible"],
                "ltr_score": round(orig_ranked["ltr_score"], 3),
                "predicted_ctr_pct": orig_ranked["predicted_ctr_pct"],
                "features": orig_features
            },
            "simulated": {
                "price": sim_prod["price"],
                "discount_pct": sim_prod["discount_pct"],
                "rating": sim_prod["rating"],
                "review_count": sim_prod["review_count"],
                "is_prime_eligible": sim_prod["is_prime_eligible"],
                "ltr_score": round(sim_ranked["ltr_score"], 3),
                "predicted_ctr_pct": sim_ranked["predicted_ctr_pct"],
                "features": sim_features
            },
            "ctr_lift_pct": round(sim_ranked["predicted_ctr_pct"] - orig_ranked["predicted_ctr_pct"], 2),
            "score_delta": round(sim_ranked["ltr_score"] - orig_ranked["ltr_score"], 3)
        }
