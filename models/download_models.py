import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "hexgrad/Kokoro-82M"
MODEL_DIR = Path("models/model")
VOICES_DIR = Path("models/voices")

def setup_directories():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        print(f"Created directory: {MODEL_DIR}")
    
    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True)
        print(f"Created directory: {VOICES_DIR}")

def download_model_files():
    print(f"Downloading model files from {REPO_ID}...")
    
    files_to_download = ["kokoro-v0_19.pth", "config.json"]
    
    try:
        all_files = list_repo_files(REPO_ID)
        if "kokoro-v1_0.pth" in all_files:
             files_to_download.append("kokoro-v1_0.pth")
             print("Detected v1.0 model. Downloading both v1.0 and v0.19.")
    except Exception as e:
        print(f"Error checking repo files: {e}, falling back to v0_19 default or let download fail if missing.")

    for filename in files_to_download:
        print(f"Downloading {filename}...")
        try:
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            # Copy from cache to our local dir
            destination = MODEL_DIR / filename
            shutil.copy2(cached_path, destination)
            print(f"Saved {filename} to {destination}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

def download_voices():
    print(f"Scanning for voice files in {REPO_ID}...")
    try:
        all_files = list_repo_files(REPO_ID)
        voice_files = [f for f in all_files if f.startswith("voices/") and f.endswith(".pt")]
        
        print(f"Found {len(voice_files)} voice files.")
        
        for file_path in voice_files:
            filename = os.path.basename(file_path)
            print(f"Downloading voice: {filename}...")
            
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=file_path)
            destination = VOICES_DIR / filename
            shutil.copy2(cached_path, destination)
            print(f"Saved {filename} to {destination}")
            
    except Exception as e:
        print(f"Error downloading voices: {e}")

if __name__ == "__main__":
    setup_directories()
    download_model_files()
    download_voices()
    print("Download process completed.")
