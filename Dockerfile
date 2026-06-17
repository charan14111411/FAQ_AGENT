FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (curl is useful for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image (pinned for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

# Copy the pinned lockfile and install with uv (installs into the system
# interpreter; no virtualenv needed inside the container)
COPY requirements.lock .
RUN uv pip install --system --no-cache -r requirements.lock

# Pre-download and cache the sentence-transformers model (all-MiniLM-L6-v2) during the build phase.
# This ensures fast container boot times and allows the container to start offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY main.py .
COPY run_migrations.py .

# Expose application port
EXPOSE 8000

# Run migrations first, then start FastAPI with Uvicorn
CMD python run_migrations.py && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
