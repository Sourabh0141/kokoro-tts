FROM python:3.11-slim-bookworm

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    espeak-ng \
    build-essential \
    cmake \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    unidic-mecab \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

# Install CPU-only PyTorch first to avoid CUDA dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install unidic for Japanese text processing
RUN pip install unidic
RUN python -m unidic download

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Note: app directory is mounted as volume in docker-compose.yml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8880"]
