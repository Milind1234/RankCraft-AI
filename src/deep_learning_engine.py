import os
import torch
torch.set_num_threads(1)
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer
from src.config import SENTENCE_TRANSFORMER_MODEL, CATALOG_EMBEDDINGS_PATH


class DeepLearningEngine:
    """Manages neural semantic representations and similarity scoring."""

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        self.device = "cpu"  # CPU is blazing fast (~5ms) for all-MiniLM-L6-v2 and avoids Apple MPS / OpenMP clashes
        print(f"[DeepLearningEngine] Initializing {model_name} on device: {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.catalog_embeddings: Optional[np.ndarray] = None
        self.product_id_to_idx: Dict[str, int] = {}

    def build_or_load_catalog_embeddings(
        self,
        catalog: List[Dict[str, Any]],
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Encodes all products in the catalog into 384-dimensional dense vectors.
        Caches on disk to enable sub-20ms inference during live search.
        """
        self.product_id_to_idx = {p["id"]: i for i, p in enumerate(catalog)}

        if CATALOG_EMBEDDINGS_PATH.exists() and not force_recompute:
            print(f"[DeepLearningEngine] Loading cached embeddings from {CATALOG_EMBEDDINGS_PATH}...")
            self.catalog_embeddings = np.load(CATALOG_EMBEDDINGS_PATH)
            if len(self.catalog_embeddings) == len(catalog):
                return self.catalog_embeddings
            print("[DeepLearningEngine] Catalog size changed, recomputing embeddings...")

        print(f"[DeepLearningEngine] Encoding {len(catalog)} products with {SENTENCE_TRANSFORMER_MODEL}...")
        texts = [p.get("semantic_text", p["title"]) for p in catalog]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        CATALOG_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CATALOG_EMBEDDINGS_PATH, embeddings)
        self.catalog_embeddings = embeddings
        print(f"[DeepLearningEngine] Saved {embeddings.shape} embeddings to disk.")
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes user search query into normalized 384-dim vector."""
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb[0]

    def compute_similarity(
        self,
        query: str,
        products: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Computes cosine similarity between user query and a list of candidate products.
        Uses cached catalog embeddings if available for blazing-fast matrix dot-product.
        """
        q_emb = self.encode_query(query)  # shape: (384,)

        if self.catalog_embeddings is not None and all(p["id"] in self.product_id_to_idx for p in products):
            indices = [self.product_id_to_idx[p["id"]] for p in products]
            candidate_embs = self.catalog_embeddings[indices]  # shape: (N, 384)
            scores = np.dot(candidate_embs, q_emb)
        else:
            texts = [p.get("semantic_text", p["title"]) for p in products]
            candidate_embs = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            scores = np.dot(candidate_embs, q_emb)

        # Rescale scores to [0.0, 1.0] range for stability
        # Cosine with normalized vectors is in [-1, 1], typically [0.1, 0.85] for queries
        rescaled = [float(max(0.0, min(1.0, (s + 0.2) / 1.2))) for s in scores]
        return rescaled
