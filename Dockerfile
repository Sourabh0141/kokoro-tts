# =============================================================================
# Kokoro TTS Service - Docker Configuration
# =============================================================================
# Multi-stage Dockerfile for containerizing the Kokoro text-to-speech service.
# Uses Python 3.11 on Debian Bookworm slim base image for minimal footprint.

# -----------------------------------------------------------------------------
# Base Stage: Python Environment Setup
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# -----------------------------------------------------------------------------
# System Dependencies Installation
# -----------------------------------------------------------------------------
# Install essential system packages required for:
# - curl: HTTP client for health checks and downloads
# - espeak-ng: Text-to-speech synthesis utility (fallback TTS)
# - build-essential, cmake: C/C++ compilation tools for native extensions
# - mecab, libmecab-dev: Japanese morphological analysis library
# - mecab-ipadic-utf8: Japanese dictionary for MeCab
# - unidic-mecab: Universal Dependencies dictionary for Japanese
RUN apt-get update && apt-get install -y \
    curl \
    espeak-ng \
    build-essential \
    cmake \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    unidic-mecab \
    && rm -rf /var/lib/apt/lists/*  # Clean up package cache to reduce image size

# -----------------------------------------------------------------------------
# Application Directory Setup
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Fast Python Package Manager
# -----------------------------------------------------------------------------
# Install uv - a fast Python package installer and resolver
RUN pip install uv

# -----------------------------------------------------------------------------
# PyTorch Installation (CPU-only)
# -----------------------------------------------------------------------------
# Install PyTorch ecosystem with CPU-only binaries to avoid CUDA dependencies
# and reduce image size. Torch is the core ML framework for the Kokoro model.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------------------
# Japanese Text Processing Setup
# -----------------------------------------------------------------------------
# Install and download UniDic dictionary for Japanese text processing
# Required by the Kokoro model for proper Japanese language support
RUN pip install unidic
RUN python -m unidic download

# -----------------------------------------------------------------------------
# Python Dependencies Installation
# -----------------------------------------------------------------------------
# Copy requirements file and install all Python dependencies using uv
# for faster and more reliable package resolution
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application Startup
# -----------------------------------------------------------------------------
# Configure the container to run the FastAPI application via the main module
# This ensures that the application uses the configuration logic defined in
# app/main.py (loading from .env, setting env vars, etc.)
CMD ["python", "-m", "app.main"]
