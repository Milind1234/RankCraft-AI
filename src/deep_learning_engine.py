"""
Pillar 3: Deep Learning Engine
Uses Quantized ONNX MiniLM (or SentenceTransformer fallback) to compute semantic similarity
dense embeddings in under 10ms with less than 50MB RAM footprint.
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import MODELS_DIR, CATALOG_EMBEDDINGS_PATH

ONNX_MODEL_PATH = MODELS_DIR / "minilm_quantized.onnx"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"


class DeepLearningEngine:
    """Manages neural semantic representations and similarity scoring with ultra-low memory."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.catalog_embeddings: Optional[np.ndarray] = None
        self.product_id_to_idx: Dict[str, int] = {}
        self.use_onnx = ONNX_MODEL_PATH.exists() and TOKENIZER_PATH.exists()

        if self.use_onnx:
            print(f"[DeepLearningEngine] Initializing ultra-lightweight ONNX runtime engine ({ONNX_MODEL_PATH.name})...")
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
            self.tokenizer.enable_truncation(max_length=128)
            self.tokenizer.enable_padding(length=128)

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(ONNX_MODEL_PATH),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.device = "onnx-cpu"
        else:
            print(f"[DeepLearningEngine] ONNX model not found, falling back to SentenceTransformer ({model_name})...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device="cpu")
            self.device = "cpu"

    def build_or_load_catalog_embeddings(
        self,
        catalog: List[Dict[str, Any]],
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Loads precomputed catalog dense embeddings for sub-10ms matrix dot-product inference.
        """
        self.product_id_to_idx = {p["id"]: i for i, p in enumerate(catalog)}

        if CATALOG_EMBEDDINGS_PATH.exists() and not force_recompute:
            print(f"[DeepLearningEngine] Loading cached embeddings from {CATALOG_EMBEDDINGS_PATH}...")
            # Load as float32 to conserve memory
            self.catalog_embeddings = np.load(CATALOG_EMBEDDINGS_PATH).astype(np.float32)
            if len(self.catalog_embeddings) == len(catalog):
                return self.catalog_embeddings
            print("[DeepLearningEngine] Catalog size changed, recomputing embeddings...")

        if not self.use_onnx:
            print(f"[DeepLearningEngine] Encoding {len(catalog)} products...")
            texts = [p.get("semantic_text", p["title"]) for p in catalog]
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
        else:
            # Batch encode with ONNX
            embeddings = np.array([self.encode_query(p.get("semantic_text", p["title"])) for p in catalog], dtype=np.float32)

        CATALOG_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CATALOG_EMBEDDINGS_PATH, embeddings)
        self.catalog_embeddings = embeddings
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes user search query into normalized 384-dim dense vector."""
        if self.use_onnx:
            encoded = self.tokenizer.encode(query)
            inputs = {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([encoded.type_ids], dtype=np.int64)
            }
            outputs = self.session.run(None, inputs)
            last_hidden_state = outputs[0]  # shape: (1, seq_len, 384)

            # Mean pooling with attention mask
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embedding = sum_embeddings / sum_mask

            # L2 normalize
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized = (embedding / np.clip(norm, a_min=1e-9, a_max=None))[0]
            return normalized.astype(np.float32)
        else:
            emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            return emb[0].astype(np.float32)

    def compute_similarity(
        self,
        query: str,
        products: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Computes cosine similarity between user query and candidate products.
        """
        q_emb = self.encode_query(query)  # shape: (384,)

        if self.catalog_embeddings is not None and all(p["id"] in self.product_id_to_idx for p in products):
            indices = [self.product_id_to_idx[p["id"]] for p in products]
            candidate_embs = self.catalog_embeddings[indices]  # shape: (N, 384)
            scores = np.dot(candidate_embs, q_emb)
        else:
            candidate_embs = np.array([self.encode_query(p.get("semantic_text", p["title"])) for p in products], dtype=np.float32)
            scores = np.dot(candidate_embs, q_emb)

        # Rescale scores to [0.0, 1.0] range for stability
        rescaled = [float(max(0.0, min(1.0, (s + 0.2) / 1.2))) for s in scores]
        return rescaled
