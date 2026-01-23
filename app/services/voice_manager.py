# =============================================================================
# Voice Manager - Lazy Loading and Memory Management for TTS Voices
# =============================================================================
# Advanced voice embedding manager that provides thread-safe lazy loading,
# automatic TTL-based memory management, and comprehensive statistics tracking.
# Optimizes memory usage by loading voices on-demand and unloading unused ones.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
import torch

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.logging import logger


@dataclass
class LoadedVoiceInfo:
    """
    Metadata container for a loaded voice embedding.

    Stores the voice tensor along with metadata for memory management,
    access tracking, and statistics reporting.

    Attributes:
        voice_tensor: The PyTorch tensor containing the voice embedding
        last_access_time: Timestamp of last access (for TTL calculations)
        language: Language name (e.g., "American English")
        voice_name: Display name of the voice (e.g., "Bella (Female)")
        voice_id: File identifier (e.g., "af_bella")
        size_mb: Memory size of the tensor in megabytes
    """

    voice_tensor: torch.Tensor
    last_access_time: datetime
    language: str
    voice_name: str
    voice_id: str
    size_mb: float


class VoiceManager:
    """
    Thread-safe voice embedding manager with automatic memory management.

    Provides lazy loading of voice tensors from disk with TTL-based cleanup,
    ensuring optimal memory usage while maintaining performance for frequently
    used voices. Uses hierarchical locking to allow concurrent loading of
    different voices while preventing duplicate loads of the same voice.

    Key Features:
    - Lazy loading: Voices loaded on-demand from disk
    - TTL expiration: Automatic unloading of unused voices (default 10 minutes)
    - Thread safety: Concurrent access with granular locking
    - Memory tracking: Detailed statistics and monitoring
    - Background cleanup: Automatic expired voice removal

    Attributes:
        voices_dir: Path to directory containing voice .pt files
        device: PyTorch device for loading tensors (cpu/cuda/mps)
        ttl_seconds: Time-to-live for loaded voices in seconds
        loaded_voices: Cache of currently loaded voice embeddings
        main_lock: Primary lock protecting the loaded_voices dictionary
        key_locks: Per-voice locks for thread-safe individual loading
        cleanup_thread: Background thread for TTL cleanup
        should_stop: Flag to signal cleanup thread shutdown
        total_unloaded: Counter of voices unloaded (for statistics)
    """

    def __init__(
        self,
        voices_dir: str,
        device: str,
        ttl_seconds: int,
        shutdown_timeout_seconds: int,
    ):
        """
        Initialize the voice manager.

        Args:
            voices_dir: Path to directory containing .pt voice files
            device: PyTorch device string ('cpu', 'cuda', 'mps')
            ttl_seconds: Time-to-live for loaded voices in seconds
            shutdown_timeout_seconds: Timeout for waiting for cleanup thread to stop
        """
        self.voices_dir = voices_dir
        self.device = device
        self.ttl_seconds = ttl_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

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
        """
        Get or create a lock for a specific voice key.

        Uses lazy initialization to create locks only when needed,
        allowing for dynamic voice loading without pre-allocating locks.

        Args:
            key: Tuple of (language, voice_name) identifying the voice

        Returns:
            threading.Lock: The lock associated with this voice key
        """
        with self.main_lock:
            if key not in self.key_locks:
                self.key_locks[key] = threading.Lock()
            return self.key_locks[key]

    def load_voice(self, language: str, voice_name: str, voice_id: str) -> torch.Tensor:
        """
        Load voice tensor from disk or return cached version.

        Implements double-checked locking pattern to ensure thread-safe lazy loading.
        Updates last access time on cache hits and performs comprehensive error handling.

        Args:
            language: Language name (e.g., "American English")
            voice_name: Voice display name (e.g., "Bella (Female)")
            voice_id: Voice file identifier (e.g., "af_bella")

        Returns:
            torch.Tensor: The loaded voice embedding tensor

        Raises:
            FileNotFoundError: If the voice file doesn't exist on disk
            Exception: If loading or processing fails
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
                # This is the slow part - load tensor from disk
                voice_tensor = torch.load(voice_path, map_location=self.device)

                # Calculate size in MB for memory tracking
                size_mb = (
                    voice_tensor.numel() * voice_tensor.element_size() / (1024 * 1024)
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

                    # Log only once per load operation
                    logger.info(
                        f"Loaded voice: {voice_name} ({voice_id}) - {size_mb:.1f}MB - {load_time:.2f}ms"
                    )

                return voice_tensor

            except Exception as e:
                logger.error(
                    f"Failed to load voice {voice_id}: {str(e)}", exc_info=True
                )
                raise Exception(f"Failed to load voice {voice_id}: {str(e)}")

    def get_voice(self, language: str, voice_name: str) -> Optional[torch.Tensor]:
        """
        Get loaded voice if available (without loading from disk).

        Updates last access time if found. This method only checks the cache
        and does not perform disk I/O, making it very fast for cache hits.

        Args:
            language: Language name (e.g., "American English")
            voice_name: Voice display name (e.g., "Bella (Female)")

        Returns:
            torch.Tensor or None: The voice tensor if loaded, None if not in cache
        """
        with self.main_lock:
            key = (language, voice_name)
            if key in self.loaded_voices:
                self.loaded_voices[key].last_access_time = datetime.now()
                return self.loaded_voices[key].voice_tensor
            return None

    def unload_voice(self, language: str, voice_name: str) -> bool:
        """
        Explicitly unload a voice from memory.

        Removes the voice from cache and deletes the tensor to free memory.
        Increments the total unloaded counter for statistics tracking.

        Args:
            language: Language name (e.g., "American English")
            voice_name: Voice display name (e.g., "Bella (Female)")

        Returns:
            bool: True if voice was unloaded, False if not found in cache
        """
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
        """
        Check all loaded voices and unload expired ones.

        Scans through all cached voices and removes those that have exceeded
        their time-to-live. Updates statistics and logs cleanup activity.

        Returns:
            int: Number of voices that were unloaded
        """
        with self.main_lock:
            now = datetime.now()
            expired = []

            # First pass: identify expired voices
            for key, info in self.loaded_voices.items():
                age = (now - info.last_access_time).total_seconds()
                if age > self.ttl_seconds:
                    expired.append(key)

            # Second pass: unload expired voices
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

    def start_cleanup_loop(self, check_interval_seconds: int):
        """
        Start background thread that periodically checks for expired voices.

        Creates and starts a daemon thread that runs the cleanup loop,
        automatically removing voices that have exceeded their TTL.
        The thread sleeps in small intervals to allow for responsive shutdown.

        Args:
            check_interval_seconds: How often to check for expired voices
        """

        def cleanup_loop():
            """Background cleanup loop function."""
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
        """
        Stop background cleanup thread and unload all voices.

        Signals the cleanup thread to stop and waits for it to terminate.
        Then forcibly unloads all remaining voices from memory to ensure
        clean shutdown and prevent memory leaks.
        """
        self.should_stop = True

        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=self.shutdown_timeout_seconds)

        # Unload all remaining voices
        with self.main_lock:
            remaining_keys = list(self.loaded_voices.keys())
            for key in remaining_keys:
                info = self.loaded_voices.pop(key)
                del info.voice_tensor
                self.total_unloaded += 1

        logger.info("Voice cleanup loop stopped")

    def get_stats(self) -> dict:
        """
        Get current statistics about loaded voices.

        Provides comprehensive information about memory usage, loaded voices,
        and detailed metadata for each cached voice.

        Returns:
            dict: Statistics containing:
                - loaded_voices: Number of currently loaded voices
                - total_memory_mb: Total memory used by all loaded voices
                - total_unloaded: Total number of voices unloaded since startup
                - voices_detail: List of detailed info for each loaded voice
        """
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
