# CPU image. For GPU, start FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime and drop the torch line.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf PLAGAI_AI_MODEL_DIR=/models/ai-detector PLAGAI_CORPUS_DIR=/data/corpus

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres]"

RUN useradd -m app && mkdir -p /models /data && chown -R app /models /data
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "plagai.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
