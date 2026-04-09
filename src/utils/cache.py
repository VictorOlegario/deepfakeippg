"""Caching utilities to avoid reprocessing videos."""

import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalCache:
    """Disk-based cache for extracted rPPG signals and features.

    Stores numpy arrays and metadata as .npz and .json files
    in a structured cache directory.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, video_path: str, stage: str) -> str:
        """Generate a unique cache key for a video + processing stage.

        Args:
            video_path: Path to the source video.
            stage: Processing stage name (e.g., 'rppg', 'features').

        Returns:
            Hex digest cache key.
        """
        content = f"{video_path}:{stage}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, video_path: str, stage: str) -> Optional[dict[str, Any]]:
        """Retrieve cached data for a video and stage.

        Args:
            video_path: Path to the source video.
            stage: Processing stage name.

        Returns:
            Dictionary with cached numpy arrays, or None if not cached.
        """
        key = self._make_key(video_path, stage)
        npz_path = self.cache_dir / f"{key}.npz"

        if not npz_path.exists():
            return None

        try:
            data = dict(np.load(str(npz_path), allow_pickle=True))
            logger.debug("Cache hit: %s [%s]", video_path, stage)
            return data
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", video_path, e)
            return None

    def put(self, video_path: str, stage: str, data: dict[str, np.ndarray]) -> None:
        """Store data in the cache.

        Args:
            video_path: Path to the source video.
            stage: Processing stage name.
            data: Dictionary of numpy arrays to cache.
        """
        key = self._make_key(video_path, stage)
        npz_path = self.cache_dir / f"{key}.npz"

        try:
            np.savez_compressed(str(npz_path), **data)
            logger.debug("Cached: %s [%s]", video_path, stage)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", video_path, e)

    def has(self, video_path: str, stage: str) -> bool:
        """Check if data exists in cache.

        Args:
            video_path: Path to the source video.
            stage: Processing stage name.

        Returns:
            True if cached data exists.
        """
        key = self._make_key(video_path, stage)
        return (self.cache_dir / f"{key}.npz").exists()

    def clear(self) -> int:
        """Remove all cached files.

        Returns:
            Number of files removed.
        """
        count = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached files", count)
        return count
