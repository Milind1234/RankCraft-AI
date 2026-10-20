"""
FastAPI Server for RankCraft-AI
Exposes REST endpoints for search re-ranking, query inspection,
CTR simulation, metrics benchmarks, and serves the luxury frontend.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.config import FRONTEND_DIR
from src.pipeline import SearchPipeline

app = FastAPI(
    title="RankCraft-AI API",
    description="E-Commerce Search Query Re-Ranking & Click-Through (CTR) Predictor",
    version="2.0.0"
)

# Enable CORS for local development and cloud deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton
pipeline: Optional[SearchPipeline] = None


@app.on_event("startup")
def startup_event():
    """Initializes models and datasets on application boot."""
    global pipeline
    print("[API] Initializing SearchPipeline...")
    pipeline = SearchPipeline()
    print("[API] RankCraft-AI Pipeline ready.")


# Pydantic Request Models
class SearchRequest(BaseModel):
    query: str = Field(..., example="lightweight running shoes with arch support")
    top_k: int = Field(default=80, ge=10, le=200)
    limit: int = Field(default=24, ge=1, le=100)


class SimulateRequest(BaseModel):
    product_id: str = Field(..., example="prod_0001")
    query: str = Field(default="lightweight running shoes with arch support")
    price_override: Optional[float] = None
    discount_override: Optional[float] = None
    rating_override: Optional[float] = None
    review_count_override: Optional[int] = None
    prime_override: Optional[bool] = None


@app.get("/api/health")
def health_check():
    """Health status and system readiness check."""
    return {
        "status": "healthy",
        "service": "RankCraft-AI",
        "device": pipeline.deep_engine.device if pipeline else "initializing",
        "catalog_size": len(pipeline.catalog) if pipeline else 0,
        "model_loaded": pipeline.ltr_engine.model is not None if pipeline else False
    }


@app.post("/api/search")
def search_products(req: SearchRequest):
    """
    Core Search & Re-Ranking Endpoint:
    Returns Tri-View results (LambdaMART LTR, Pure Semantic, BM25) + NLP Query Breakdown.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine is initializing.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = pipeline.search_and_rerank(
        query=req.query,
        top_k_candidates=req.top_k,
        display_limit=req.limit
    )
    return results


@app.post("/api/simulate")
def simulate_ctr(req: SimulateRequest):
    """
    Interactive CTR Sandbox:
    Dynamically recalculates LambdaMART score and predicted CTR % upon feature adjustments.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine is initializing.")

    result = pipeline.simulate_what_if(
        product_id=req.product_id,
        query=req.query,
        price_override=req.price_override,
        discount_override=req.discount_override,
        rating_override=req.rating_override,
        review_count_override=req.review_count_override,
        prime_override=req.prime_override
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/metrics")
def get_metrics():
    """
    Retrieves NDCG@5, NDCG@10, MRR, and MAP benchmark comparisons and lifts.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine is initializing.")
    return pipeline.ltr_engine.evaluation_results


@app.get("/api/features/importance")
def get_feature_importance():
    """
    Retrieves global feature importance attribution for LambdaMART.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine is initializing.")
    return pipeline.ltr_engine.evaluation_results.get("feature_importances", [])


@app.get("/api/catalog/sample")
def get_sample_catalog(limit: int = 12):
    """Returns random sample of products for quick exploration."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine is initializing.")
    sample = pipeline.catalog[:limit]
    return {"products": sample}


# Mount Static Frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend not found at frontend/index.html"}

    @app.get("/{full_path:path}")
    def serve_static_fallback(full_path: str):
        # Allow API routes to be handled first
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="File not found")
