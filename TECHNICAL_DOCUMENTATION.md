# Technical Documentation: Kokoro TTS Service

**Generated Date:** 2026-01-23
**Version:** 1.0.0

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Request Flow](#4-request-flow)
5. [Environment & Configuration](#5-environment--configuration)
6. [Quickstart (Local & Docker)](#6-quickstart-local--docker)
7. [API Reference](#7-api-reference)
8. [Data & Persistence](#8-data--persistence)
9. [Security](#9-security)
10. [Observability](#10-observability)
11. [Testing & Quality](#11-testing--quality)
12. [Known Gaps & Roadmap](#12-known-gaps--roadmap)

***

## 1. Overview

The **Kokoro TTS Service** is a high-performance, containerized Text-to-Speech (TTS) API built with **FastAPI** and the **Kokoro-82M** model. It provides real-time audio synthesis with support for multiple languages and voices. The service features lazy model loading, thread-safe voice management with TTL-based memory cleanup, and a production-ready Docker deployment.

**Key Features:**

* **FastAPI Backend:** Async API with OpenAPI documentation.
* **Kokoro-82M Engine:** High-quality neural TTS.
* **Dynamic Voice Loading:** Lazy loading of voices to optimize memory usage.
* **Memory Management:** Automatic TTL-based cleanup of unused voice embeddings.
* **Containerization:** Optimized multi-stage Dockerfile (Debian Bookworm based).

***

## 2. Architecture

### High-Level Components

```mermaid
graph TD
    Client[Client Request] -->|HTTP/HTTPS| LB[Load Balancer / Reverse Proxy]
    LB -->|Port 8880| Uvicorn[Uvicorn Server]
    Uvicorn --> FastAPI[FastAPI App]
    
    subgraph "Application Core"
        FastAPI --> Auth[API Key Auth]
        FastAPI --> Router[API Router V1]
        Router --> Endpoint[Endpoints /audio, /voices]
        Endpoint --> TTSEngine[TTS Service Engine]
        
        TTSEngine --> VoiceMgr[Voice Manager]
        TTSEngine --> Model[Kokoro Model (PyTorch)]
    end
    
    subgraph "Filesystem / Storage"
        VoiceMgr -->|Load .pt| VoicesDir[./models/voices/*.pt]
        Model -->|Load .pth| ModelDir[./models/model/*.pth]
    end
```

### Components Description

1. **Entrypoint:** `Uvicorn` serves the ASGI application.
2. **API Layer:** `FastAPI` handles routing, validation (Pydantic), and middleware (request logging).
3. **Service Layer:**
   * `TTSEngine`: Orchestrates the generation pipeline, thread pool execution for blocking inference, and model lifecycle.
   * `VoiceManager`: Handles thread-safe loading of voice embeddings from disk and manages memory using a TTL (Time-To-Live) strategy.
4. **Inference:** Uses `torch` and `kokoro` libraries for audio synthesis.

***

## 3. Project Structure

```text
D:\Projects\text_to_speech\
├── .env.example            # Template for environment variables
├── docker-compose.yml      # Orchestration for local/dev usage
├── Dockerfile              # Multi-stage build for production
├── requirements.txt        # Python dependencies (pinned versions)
├── app/
│   ├── __init__.py
│   ├── main.py             # App entrypoint, lifespan manager, middleware
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints.py # Route handlers (/audio, /health, /voices)
│   │       └── router.py    # Router configuration
│   ├── core/
│   │   ├── config.py       # Pydantic Settings & env var loading
│   │   ├── dependencies.py # DI (Engine singleton) & Auth (API Key)
│   │   ├── exceptions.py   # Custom exception classes
│   │   └── logging.py      # Structured logging setup
│   ├── schemas/
│   │   └── tts.py          # Pydantic models for API I/O
│   └── services/
│       ├── tts.py          # Core engine logic (Kokoro wrapper)
│       └── voice_manager.py # Voice loading & memory management
└── models/
    ├── download_models.py  # Script to fetch weights from HuggingFace
    ├── model/              # Directory for model weights (.pth)
    └── voices/             # Directory for voice embeddings (.pt)
```

***

## 4. Request Flow

**Scenario:** Client requests audio generation (`POST /v1/audio`).

1. **Auth:** `api_key_auth` dependency checks `X-API-Key` header against `SERVICE__API_KEY`.
2. **Validation:** `TTSRequest` schema validates input (text length, supported language check performed later).
3. **Routing:** `app/api/v1/endpoints.py` -> `generate_audio`.
4. **Service Call:** `TTSEngine.generate()` is called (wrapped in `asyncio.to_thread` to prevent blocking the event loop).
5. **Voice Resolution:** `VoiceManager` checks memory cache.
   * *Hit:* Returns cached tensor, updates `last_access_time`.
   * *Miss:* Loads `.pt` file from disk, caches it, returns tensor.
6. **Inference:** `KPipeline` processes text -> `KModel` generates audio -> `soundfile` encodes to WAV.
7. **Response:** Returns `Response` object with WAV bytes and `audio/wav` media type.

***

## 5. Environment & Configuration

Configuration is managed via **Pydantic Settings** (`app/core/config.py`). Variables can be set via `.env` file or shell environment variables. Double underscores (`__`) denote nesting.

### Critical Environment Variables

| Variable Name | Purpose | Type | Default | Sensitive? |
| :--- | :--- | :--- | :--- | :--- |
| **Service** | | | | |
| `SERVICE__API_KEY` | API Authentication Key | String | `dev_api_key` | **YES** |
| `SERVICE__PORT` | HTTP Port | Integer | `8880` | No |
| `SERVICE__ENVIRONMENT`| Env name (dev/prod) | String | `development` | No |
| `SERVICE__LOG_LEVEL` | Logging verbosity | String | `INFO` | No |
| **Model** | | | | |
| `MODEL__DEVICE` | Inference device (`cpu`, `cuda`, `mps`) | String | `cpu` | No |
| `MODEL__DTYPE` | Precision (`fp32`, `fp16`) | String | `fp32` | No |
| **Limits** | | | | |
| `LIMITS__MAX_TEXT_LENGTH` | Max chars per request | Integer | `5000` | No |
| **Cache** | | | | |
| `CACHE__TTL_SECONDS` | Voice memory TTL | Integer | `600` (10m) | No |

### Example .env (Redacted)

```bash
# Security
SERVICE__API_KEY=<REDACTED>  # Change this in production!

# Compute
MODEL__DEVICE=cpu            # Use 'cuda' for GPU
MODEL__DTYPE=fp32            # Use 'fp16' for GPU

# Application
SERVICE__HOST=0.0.0.0
SERVICE__PORT=8880
```

***

## 6. Quickstart (Local & Docker)

### Prerequisites

* Python 3.11+
* Docker & Docker Compose (optional but recommended)

### Option A: Local Development

1. **Clone and Setup:**
   ```bash
   git clone <repo_url>
   cd text_to_speech
   python -m venv venv

   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Download Models:**
   **Crucial Step:** You must download the model weights and voices before starting.
   ```bash
   python models/download_models.py
   ```

3. **Configuration:**
   Copy the example env file and (optionally) edit it.
   ```bash
   cp .env.example .env
   ```

4. **Run Server:**
   ```bash
   python -m app.main
   # OR directly with uvicorn
   uvicorn app.main:app --host 0.0.0.0 --port 8880 --reload
   ```

5. **Verify:**
   Open `http://localhost:8880/docs` to see the Swagger UI.

### Option B: Docker (Production Recommended)

1. **Build and Run:**
   ```bash
   # Downloads models if not present, but better to map them via volume
   # First, run the downloader locally to populate ./models
   python models/download_models.py

   # Start container
   docker-compose up --build -d
   ```

2. **Check Logs:**
   ```bash
   docker-compose logs -f
   ```

***

## 7. API Reference

**Base URL:** `/v1`
**Auth:** Header `X-API-Key: <your_key>`

### 1. Generate Audio

* **POST** `/audio`
* **Description:** Converts text to speech.
* **Body:**
  ```json
  {
    "text": "Hello world, this is a test.",
    "language": "American English",
    "voice": "Bella (Female)",
    "speed": 1.0
  }
  ```
* **Response:** Binary WAV file (`audio/wav`).

### 2. List Voices

* **GET** `/voices`
* **Description:** Returns available languages and voices.
* **Response:**
  ```json
  {
    "languages": {
      "American English": {
        "Bella (Female)": "af_bella",
        "Adam (Male)": "am_adam"
      }
    },
    "total_voices": 2,
    "total_languages": 1
  }
  ```

### 3. Health Check

* **GET** `/health`
* **Description:** Returns service health and memory usage.
* **Response:**
  ```json
  {
    "status": "ok",
    "loaded_voices": 1,
    "memory_usage_mb": 150.5,
    "is_ready": true
  }
  ```

***

## 8. Data & Persistence

### Database

* **None.** This service is stateless.

### Filesystem

* **Models:** Stored in `models/model/` (e.g., `kokoro-v1_0.pth`).
* **Voices:** Stored in `models/voices/` (e.g., `af_bella.pt`).
* **Persistence:** The `models/` directory is mounted as a volume in Docker (`./models:/app/models:ro`) to persist downloaded weights and avoid re-downloading on container restart.

***

## 9. Security

### Authentication

* **Mechanism:** API Key.
* **Implementation:** `app.core.dependencies.api_key_auth`.
* **Best Practice:** The key is passed via `X-API-Key` header.
* **Recommendation:** Rotate keys periodically. In production, consider placing this service behind an API Gateway (Kong, Nginx) that handles more complex auth (OAuth2/OIDC) and rate limiting.

### Secrets Management

* Secrets (API Key) are read from `.env` or environment variables.
* **Do not commit `.env` to Git.**
* In Kubernetes, use `Secret` objects mapped to environment variables.
* In Docker, use `env_file` (excluded from git) or pass variables at runtime.

### Input Validation

* Strict validation is enforced via Pydantic (`app/schemas/tts.py`).
* `text` is limited to `LIMITS__MAX_TEXT_LENGTH` (default 5000) to prevent DoS attacks via memory exhaustion.

***

## 10. Observability

### Logging

* **Format:** Structured console logs (timestamp, level, logger name).
* **Configuration:** `app/core/logging.py` reduces noise from `transformers` and `torch`.
* **Access Logs:** Custom middleware in `main.py` logs `METHOD PATH STATUS DURATION`.

### Metrics

* **Current State:** The `/health` endpoint exposes basic internal metrics (memory usage, loaded voices).
* **Missing:** No Prometheus/OpenTelemetry integration.

***

## 11. Testing & Quality

### Current Status

* **Tests:** **NONE**. The repository currently lacks a test suite.
* **CI/CD:** **NONE**.

### Recommended Test Plan

1. **Unit Tests:** Test `VoiceManager` logic (loading, TTL expiry) and `TTSEngine` validation logic.
2. **Integration Tests:** Test FastAPI endpoints using `TestClient` (mocking the heavy `kokoro` model to avoid slow inference during tests).

### Example `pytest` Skeleton

Create `tests/conftest.py` and `tests/test_api.py`:

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_auth_missing():
    response = client.post("/v1/audio", json={"text": "test", "language": "a", "voice": "b"})
    assert response.status_code == 401
```

### Recommended CI Workflow (GitHub Actions)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt pytest httpx
      - run: pytest
```

***

## 12. Known Gaps & Roadmap

| Gap | Impact | Remediation |
| :--- | :--- | :--- |
| **No Tests** | High risk of regression. | Add `pytest` suite immediately. |
| **No CI/CD** | Manual deployment risk. | Add GitHub Actions workflow. |
| **Rate Limiting** | Risk of abuse/DoS. | Implement `slowapi` or use external gateway. |
| **Metric Export** | Hard to monitor in prod. | Add `prometheus-fastapi-instrumentator`. |
| **GPU Support** | Dockerfile is CPU-only. | Create `Dockerfile.gpu` with CUDA base image. |

***
