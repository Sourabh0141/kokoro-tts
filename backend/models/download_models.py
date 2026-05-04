# =============================================================================
# Kokoro TTS Model and Voice Downloader
# =============================================================================
# This script downloads the Kokoro-82M model files and voice embeddings from
# Hugging Face Hub for use with the Kokoro text-to-speech service.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import os
import shutil
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from huggingface_hub import hf_hub_download, list_repo_files

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
# Hugging Face repository containing the Kokoro model and voices
REPO_ID = "hexgrad/Kokoro-82M"

# Local directories for storing downloaded files
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"  # Directory for model weights and config
VOICES_DIR = BASE_DIR / "voices"  # Directory for voice embedding files


def setup_directories():
    """
    Create necessary directories for model and voice storage.

    Creates the models/model and models/voices directories if they don't exist.
    These directories are required for storing the downloaded Kokoro model files
    and voice embeddings respectively.
    """
    print("Setting up directories...")

    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        print(f"Created directory: {MODEL_DIR}")

    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True)
        print(f"Created directory: {VOICES_DIR}")

    print("Directory setup complete.")


def download_model_files():
    """
    Download Kokoro model files from Hugging Face Hub.

    Downloads the core model weights (kokoro-v0_19.pth or kokoro-v1_0.pth)
    and configuration file (config.json). Automatically detects and downloads
    the latest available model version.

    Files are downloaded to the models/model directory.
    """
    print(f"Downloading model files from {REPO_ID}...")

    # Base files that are always needed
    files_to_download = ["kokoro-v0_19.pth", "config.json"]

    # Check for newer model version (v1.0)
    try:
        all_files = list_repo_files(REPO_ID)
        if "kokoro-v1_0.pth" in all_files:
            files_to_download.append("kokoro-v1_0.pth")
            print("Detected v1.0 model. Downloading both v1.0 and v0.19.")
    except Exception as e:
        print(f"Warning: Could not check for model versions: {e}")
        print("Proceeding with default model files.")

    # Download each model file
    for filename in files_to_download:
        print(f"Downloading {filename}...")
        try:
            # Download from Hugging Face Hub to local cache
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)

            # Copy from cache to our local model directory
            destination = MODEL_DIR / filename
            shutil.copy2(cached_path, destination)
            print(f"Saved {filename} to {destination}")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print("Continuing with other files...")

    print("Model file download process completed.")


def download_voices():
    """
    Download all available voice embedding files from Hugging Face Hub.

    Scans the repository for voice files (located in voices/ directory with .pt extension),
    then downloads each voice embedding file. Voice files contain the learned
    speaker characteristics used by the Kokoro model.

    Files are downloaded to the models/voices directory.
    """
    print(f"Scanning for voice files in {REPO_ID}...")

    try:
        # Get list of all files in the repository
        all_files = list_repo_files(REPO_ID)

        # Filter for voice embedding files (.pt files in voices/ directory)
        voice_files = [f for f in all_files if f.startswith("voices/") and f.endswith(".pt")]

        print(f"Found {len(voice_files)} voice files.")

        # Download each voice file
        for file_path in voice_files:
            filename = os.path.basename(file_path)
            print(f"Downloading voice: {filename}...")

            try:
                # Download from Hugging Face Hub
                cached_path = hf_hub_download(repo_id=REPO_ID, filename=file_path)

                # Copy to local voices directory
                destination = VOICES_DIR / filename
                shutil.copy2(cached_path, destination)
                print(f"Saved {filename} to {destination}")

            except Exception as e:
                print(f"Failed to download voice {filename}: {e}")
                print("Continuing with other voices...")

    except Exception as e:
        print(f"Error downloading voices: {e}")
        print("Voice download failed. Model files may still be usable.")

    print("Voice download process completed.")


if __name__ == "__main__":
    """
    Main execution entry point.

    Runs the complete model and voice download process:
    1. Creates necessary directories
    2. Downloads model files (weights and config)
    3. Downloads all available voice embeddings

    This script should be run before starting the TTS service to ensure
    all required model files are available locally.
    """
    print("Starting Kokoro model and voice download process...")
    print("=" * 60)

    setup_directories()
    print()

    download_model_files()
    print()

    download_voices()
    print()

    print("=" * 60)
    print("Download process completed.")
    print("You can now start the TTS service.")
