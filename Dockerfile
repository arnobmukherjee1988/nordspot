# ── SE3 Electricity Price Forecast — App Image ────────────────────────────────
# Runs the Streamlit dashboard.
# Training is done separately:
#   docker-compose run --rm app python -m ml.train
#
# Build:  docker build -t se3-app .
# Run:    docker-compose up
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# System dependencies
# libgomp1  — required by LightGBM (OpenMP multi-threading)
# curl      — used by the container health check
# git       — required by pip to install timedb directly from GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (everything except what .dockerignore excludes)
COPY db/        db/
COPY ml/        ml/
COPY pipeline/  pipeline/
COPY dashboard/ dashboard/

# model/ is NOT baked in — it is bind-mounted at runtime via docker-compose.
# This placeholder ensures the directory exists if the mount is missing.
RUN mkdir -p model

# Entrypoint script: downloads model artifacts from S3 if model/ is empty
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

# Liveness probe — Streamlit exposes a health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# Run the dashboard on all interfaces so Docker port-mapping works
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
