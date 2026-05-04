# =============================================================================
# Dockerfile for Streamlit Frontend
# =============================================================================
#
# This Dockerfile creates a container for the Streamlit frontend of the
# Kokoro TTS application.
#
# It uses a multi-stage build to keep the final image size small.
#
# Stage 1: Build Stage
# - Uses the official Python image.
# - Installs dependencies into a virtual environment.
#
# Stage 2: Final Stage
# - Uses a slim Python image.
# - Copies the virtual environment from the build stage.
# - Copies the application code.
# - Sets the entrypoint to run the Streamlit app.
#
# =============================================================================

# --- Stage 1: Build Stage ---
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
# --no-cache-dir: Disables the cache to reduce image size.
# --upgrade pip: Ensures the latest version of pip is used.
# -r requirements.txt: Installs all dependencies from the requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Stage ---
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Set the entrypoint to run the Streamlit application
# --server.port: The port to run the server on.
# --server.address: The address to bind the server to (0.0.0.0 for Docker).
# --server.headless: Runs Streamlit in headless mode (recommended for Docker).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
