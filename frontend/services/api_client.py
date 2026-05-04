# =============================================================================
# API Client for Kokoro TTS Service
# =============================================================================
# This module provides a client for interacting with the backend Kokoro TTS API.
# It handles request authentication, sending, and response/error processing.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import logging
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
import requests

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from config import AppConfig

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Class Definition
# -----------------------------------------------------------------------------
class TTSClient:
    """
    A client for making requests to the Kokoro TTS backend API.

    This class encapsulates the logic for sending HTTP requests to the various
    endpoints of the TTS service, including health checks, voice listing, and
    audio generation. It automatically handles authentication via API key.

    Attributes:
        config (AppConfig): The application configuration settings.
        base_url (str): The constructed base URL for API v1 endpoints.
        headers (dict): The default headers for all outgoing requests.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the TTSClient with the application configuration.

        Args:
            config (AppConfig): An instance of the application settings.
        """
        self.config = config
        self.base_url = config.API_URL
        self.headers = {
            "X-API-Key": config.API_KEY,
            "Content-Type": "application/json",
        }

    def _handle_response(self, response: requests.Response) -> Any:
        """
        Processes HTTP responses, handling both success and error cases.

        Args:
            response (requests.Response): The HTTP response object.

        Returns:
            Any: The JSON content if the response is JSON, otherwise the raw content.

        Raises:
            Exception: If the HTTP request resulted in an error status code.
        """
        try:
            response.raise_for_status()
            if "application/json" in response.headers.get("content-type", ""):
                return response.json()
            return response.content
        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error: {e}"
            if "application/json" in response.headers.get("content-type", ""):
                try:
                    detail = response.json().get("detail", "No details provided.")
                    error_msg = f"API Error: {detail}"
                except ValueError:
                    # The response was not valid JSON
                    pass
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise Exception(f"An unexpected error occurred: {e}") from e

    def get_health(self) -> Dict[str, Any]:
        """
        Fetches the health status of the backend API.

        Returns:
            A dictionary containing the health status, or an error message
            if the request fails.
        """
        try:
            url = f"{self.base_url}health"
            response = requests.get(url, headers=self.headers, timeout=5)
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "detail": str(e)}

    def get_voices(self) -> Dict[str, Any]:
        """
        Fetches the list of available voices from the API.

        Returns:
            A dictionary of available languages and voices.

        Raises:
            Exception: If the request to the voices endpoint fails.
        """
        url = f"{self.base_url}voices"
        response = requests.get(url, headers=self.headers, timeout=10)
        return self._handle_response(response)

    def generate_audio(
        self, text: str, voice: str, language: str, speed: float
    ) -> bytes:
        """
        Requests audio generation from the API.

        Args:
            text (str): The text to be synthesized.
            voice (str): The selected voice.
            language (str): The selected language.
            speed (float): The desired speech speed.

        Returns:
            The generated audio content in bytes.

        Raises:
            Exception: If the audio generation request fails.
        """
        url = f"{self.base_url}audio"
        payload = {
            "text": text,
            "voice": voice,
            "language": language,
            "speed": speed,
        }

        response = requests.post(url, json=payload, headers=self.headers, stream=True)
        return self._handle_response(response)
