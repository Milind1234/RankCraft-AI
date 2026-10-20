"""
Entry point for RankCraft-AI
Starts the FastAPI application and opens the luxury frontend.
"""
import uvicorn
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RankCraft-AI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)), help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 RankCraft-AI: E-Commerce Search Query Re-Ranking & CTR Predictor")
    print(f"📡 Server running on http://{args.host}:{args.port}")
    print("✨ Serving luxury web frontend and REST API endpoints")
    print("=" * 70)

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
