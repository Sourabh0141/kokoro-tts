FROM python:3.10-slim-bookworm

# System deps
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Requirements & Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Config and Source
COPY kokoro_tts ./kokoro_tts
COPY app ./app

# Set PYTHONPATH
ENV PYTHONPATH=/app:/app/kokoro_tts

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8880"]
