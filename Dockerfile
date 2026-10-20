# ==============================================================================
# RankCraft-AI: Multi-Stage Search Re-Ranking & CTR Predictor
# Optimized for Ultra-Low Memory (<200MB) Free Deployment on Render & Cloud Hosts
# ==============================================================================
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

WORKDIR $HOME/app

# Copy requirements and install lightweight dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and precomputed embeddings
COPY --chown=user:user . .

# Expose default port
EXPOSE 7860

# Launch application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
