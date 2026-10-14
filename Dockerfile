# Small production image: the API with the lightweight defaults (hashing embedder + extractive answers).
# Point RAG_LLM_URL at a vLLM server and set RAG_EMBEDDER to a sentence-transformers model for real quality
# (and build with --build-arg EXTRAS=ml to include those libraries).
FROM python:3.12-slim AS base
ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install dependencies before copying the code, so code edits don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".${EXTRAS:+[$EXTRAS]}"

COPY data/sample ./data/sample
# Run as an unprivileged user: a compromised process can't modify the image.
RUN useradd --create-home app
USER app
ENV RAG_DATA_DIR=/app/data/sample
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "ragservice.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
