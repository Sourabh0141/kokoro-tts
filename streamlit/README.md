# Kokoro TTS Playground (Streamlit Frontend)

![Streamlit App](https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge\&logo=streamlit)

This directory contains the Streamlit web interface for the **Kokoro TTS** service. It provides a user-friendly playground to interact with the text-to-speech engine, allowing users to easily generate audio from text using various languages and voices.

## Features

* **Intuitive UI**: A clean and simple interface for text-to-speech conversion.
* **Dynamic Voice Selection**: Automatically populates language and voice options from the backend API.
* **Real-time System Status**: A sidebar widget shows the real-time health and resource usage of the backend service.
* **Adjustable Speech Speed**: Control the playback speed of the generated audio.
* **Audio Playback & Download**: Listen to the generated audio directly in the browser and download it as a `.wav` file.
* **Responsive Design**: Works on both desktop and mobile browsers.
* **Containerized**: Includes `Dockerfile` and `docker-compose.yml` for easy deployment with Docker.

## Prerequisites

Before running this frontend, you must have the main **Kokoro TTS backend service** running. Please see the [main project README](../../README.md) for instructions on how to set it up.

## Installation and Setup

1. **Navigate to the `streamlit` directory**:
   ```bash
   cd streamlit
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**:
   * Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   * Edit the `.env` file to match your backend configuration. The most important variable is `API_BASE_URL`, which must point to your running Kokoro TTS backend.

## Running the Application

You can run the Streamlit frontend either directly with Python or using Docker.

### 1. Running Locally with Python

Ensure your virtual environment is activated and you are in the `streamlit` directory.

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### 2. Running with Docker

This is the recommended method for a production-like setup, as it handles networking with the backend service seamlessly.

**Prerequisites**:

* Docker and Docker Compose are installed.
* The backend service is running via Docker Compose on a network named `kokoro-tts-network` (as configured in the root `docker-compose.yml`).

1. **Build and run the container**:
   From the `streamlit` directory, run:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   The application will be available at `http://localhost:8501`. The Docker container will automatically be able to communicate with the backend service container over the shared Docker network.

## Configuration

The application is configured using environment variables defined in the `.env` file.

| Variable                | Description                                                               | Default Value               |
| ----------------------- | ------------------------------------------------------------------------- | --------------------------- |
| `API_BASE_URL`          | The base URL of the backend TTS service.                                  | `http://127.0.0.1:8880`     |
| `API_KEY`               | The API key to authenticate with the backend.                             | `dev_api_key`               |
| `API_V1_PREFIX`         | The API version prefix.                                                   | `/v1/`                      |
| `APP_TITLE`             | The title of the Streamlit application.                                   | `Kokoro TTS Playground`     |
| `HEALTH_CHECK_TTL`      | Cache duration (in seconds) for the system health status.                 | `60`                        |
| `VOICES_TTL`            | Cache duration (in seconds) for the available voices list.                | `3600`                      |
| `MAX_TEXT_LENGTH`       | Maximum number of characters allowed in the text input area.              | `5000`                      |

## Connecting to the Backend

* **When running locally**: Ensure `API_BASE_URL` in your `.env` file points to the backend's address (e.g., `http://localhost:8880`).
* **When running with Docker**: The `docker-compose.yml` file is configured to connect to the backend service over a shared network named `kokoro-tts-network`. To make this work, you must update `API_BASE_URL` in the `.env` file to point to the backend container's service name:
  ```
  API_BASE_URL=http://kokoro-tts-app:8880
  ```
  This allows the Streamlit container to resolve and communicate with the backend container by its service name.
