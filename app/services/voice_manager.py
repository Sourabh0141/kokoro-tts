"""
Voice Loading Manager

Handles lazy loading of voice embeddings with TTL-based automatic unloading.
Provides thread-safe voice caching and memory management.
"""

import os

import threading

import time

from dataclasses import dataclass

from datetime import datetime

from typing import Dict, Optional, Tuple



import torch

from app.core.logging import logger



@dataclass

class LoadedVoiceInfo:

    """Metadata and tensor for a loaded voice"""

    voice_tensor: torch.Tensor  # The actual voice embedding

    last_access_time: datetime  # Updated on each access

    language: str  # Language name

    voice_name: str  # Voice display name

    voice_id: str  # Voice file ID (e.g., 'af_bella')

    size_mb: float  # Tensor size in MB





class VoiceManager:

    """

    Manages lazy loading and unloading of voice embeddings.

    

    Features:

    - On-demand loading from disk

    - Automatic TTL-based unloading

    - Thread-safe access with granular locking

    - Memory tracking and statistics

    """



    def __init__(self, voices_dir: str, device: str, ttl_seconds: int = 600):

        """

        Initialize voice manager.



        Args:

            voices_dir: Path to directory containing voice .pt files

            device: Torch device (cpu, cuda, mps)

            ttl_seconds: Time-to-live for voices (default 10 min = 600 seconds)

        """

        self.voices_dir = voices_dir

        self.device = device

        self.ttl_seconds = ttl_seconds



        # Cache of loaded voices: (language, voice_name) -> LoadedVoiceInfo

        self.loaded_voices: Dict[Tuple[str, str], LoadedVoiceInfo] = {}



        # Thread synchronization

        # main_lock protects the loaded_voices dict structure and key_locks dict

        self.main_lock = threading.RLock()

        

        # Per-voice locks to allow concurrent loading of different voices

        # key -> threading.Lock

        self.key_locks: Dict[Tuple[str, str], threading.Lock] = {}



        # Background cleanup thread

        self.cleanup_thread: Optional[threading.Thread] = None

        self.should_stop = False

        

        # Statistics

        self.total_unloaded = 0



    def _get_voice_lock(self, key: Tuple[str, str]) -> threading.Lock:

        """Get or create a lock for a specific voice key."""

        with self.main_lock:

            if key not in self.key_locks:

                self.key_locks[key] = threading.Lock()

            return self.key_locks[key]



    def load_voice(

        self, language: str, voice_name: str, voice_id: str

    ) -> torch.Tensor:

        """

        Load voice tensor from disk or return cached version.

        Updates last access time.

        

        Uses per-voice locking to allow concurrent loading of different voices.

        """

        key = (language, voice_name)

        start_time = time.time()



        # 1. Fast path: Check if already loaded (using main lock for dict access)

        with self.main_lock:

            if key in self.loaded_voices:

                self.loaded_voices[key].last_access_time = datetime.now()

                return self.loaded_voices[key].voice_tensor



        # 2. Acquire specific lock for this voice

        voice_lock = self._get_voice_lock(key)

        

        with voice_lock:

            # 3. Double-check locking pattern

            with self.main_lock:

                if key in self.loaded_voices:

                    self.loaded_voices[key].last_access_time = datetime.now()

                    return self.loaded_voices[key].voice_tensor



            # 4. Heavy load from disk (outside main_lock, inside voice_lock)

            voice_path = os.path.join(self.voices_dir, f"{voice_id}.pt")



            if not os.path.exists(voice_path):

                logger.error(f"Voice file not found: {voice_path}")

                raise FileNotFoundError(f"Voice file not found: {voice_path}")



            try:

                # This is the slow part

                voice_tensor = torch.load(voice_path, map_location=self.device)



                # Calculate size in MB

                size_mb = voice_tensor.numel() * voice_tensor.element_size() / (

                    1024 * 1024

                )

                

                load_time = (time.time() - start_time) * 1000



                # Store in cache (needs main_lock again)

                with self.main_lock:

                    info = LoadedVoiceInfo(

                        voice_tensor=voice_tensor,

                        last_access_time=datetime.now(),

                        language=language,

                        voice_name=voice_name,

                        voice_id=voice_id,

                        size_mb=size_mb,

                    )

                    self.loaded_voices[key] = info

                    

                    # Log only once

                    logger.info(f"Loaded voice: {voice_name} ({voice_id}) - {size_mb:.1f}MB - {load_time:.2f}ms")



                return voice_tensor



            except Exception as e:

                logger.error(f"Failed to load voice {voice_id}: {str(e)}", exc_info=True)

                raise Exception(f"Failed to load voice {voice_id}: {str(e)}")



    def get_voice(self, language: str, voice_name: str) -> Optional[torch.Tensor]:

        """

        Get loaded voice if available (without loading from disk).

        Updates last access time if found.

        """

        with self.main_lock:

            key = (language, voice_name)

            if key in self.loaded_voices:

                self.loaded_voices[key].last_access_time = datetime.now()

                return self.loaded_voices[key].voice_tensor

            return None



    def unload_voice(self, language: str, voice_name: str) -> bool:

        """Explicitly unload a voice from memory."""

        with self.main_lock:

            key = (language, voice_name)

            if key in self.loaded_voices:

                info = self.loaded_voices.pop(key)

                # Delete tensor to free memory

                del info.voice_tensor

                self.total_unloaded += 1

                

                logger.info(f"Unloaded voice: {voice_name} ({info.voice_id})")

                return True

            return False



    def cleanup_expired_voices(self) -> int:

        """Check all loaded voices and unload expired ones."""

        with self.main_lock:

            now = datetime.now()

            expired = []



            for key, info in self.loaded_voices.items():

                age = (now - info.last_access_time).total_seconds()

                if age > self.ttl_seconds:

                    expired.append(key)



            for key in expired:

                info = self.loaded_voices.pop(key)

                del info.voice_tensor

                age = (now - info.last_access_time).total_seconds()

                self.total_unloaded += 1

                logger.info(

                    f"TTL expired - Unloaded: {info.voice_name} ({info.voice_id}, "

                    f"age: {age:.1f}s)"

                )



            return len(expired)



    def start_cleanup_loop(self, check_interval_seconds: int = 5):

        """Start background thread that periodically checks for expired voices."""

        def cleanup_loop():

            while not self.should_stop:

                try:

                    expired_count = self.cleanup_expired_voices()

                    if expired_count > 0:

                        stats = self.get_stats()

                        logger.info(

                            f"Memory status: {stats['loaded_voices']} voices, "

                            f"{stats['total_memory_mb']:.1f}MB"

                        )

                except Exception as e:

                    logger.error(f"Error in cleanup loop: {e}", exc_info=True)



                # Sleep in small intervals so shutdown is responsive

                for _ in range(check_interval_seconds):

                    if self.should_stop:

                        break

                    time.sleep(1)



        self.should_stop = False

        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)

        self.cleanup_thread.start()

        logger.info(

            f"Voice cleanup loop started (TTL: {self.ttl_seconds}s, "

            f"check interval: {check_interval_seconds}s)"

        )



    def stop_cleanup_loop(self):

        """Stop background cleanup thread and unload all voices"""

        self.should_stop = True

        if self.cleanup_thread and self.cleanup_thread.is_alive():

            self.cleanup_thread.join(timeout=5)



        # Unload all remaining voices

        with self.main_lock:

            remaining_keys = list(self.loaded_voices.keys())

            for key in remaining_keys:

                info = self.loaded_voices.pop(key)

                del info.voice_tensor

                self.total_unloaded += 1



        logger.info("Voice cleanup loop stopped")



    def get_stats(self) -> dict:

        """Get current statistics about loaded voices."""

        with self.main_lock:

            total_memory = sum(info.size_mb for info in self.loaded_voices.values())

            return {

                "loaded_voices": len(self.loaded_voices),

                "total_memory_mb": total_memory,

                "total_unloaded": self.total_unloaded,

                "voices_detail": [

                    {

                        "language": info.language,

                        "voice": info.voice_name,

                        "voice_id": info.voice_id,

                        "size_mb": info.size_mb,

                        "age_seconds": (

                            datetime.now() - info.last_access_time

                        ).total_seconds(),

                    }

                    for info in self.loaded_voices.values()

                ],

            }
