# ==============================================================================
# RankCraft-AI: Multi-Stage Search Re-Ranking & CTR Predictor
# Optimized for 100% Free Deployment on Hugging Face Spaces & Render
# ==============================================================================
FROM python:3.11-slim

# Install system dependencies including OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (Required for Hugging Face Spaces Docker SDK)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

WORKDIR $HOME/app

# Copy requirements and install python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Sentence Transformer model to cache at build time for sub-3s boot
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source code and data assets
COPY --chown=user:user . .

# Set Hugging Face offline mode so it never pings remote hub at runtime
ENV HF_HUB_OFFLINE=1

# Expose default port
EXPOSE 7860

# Launch application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
