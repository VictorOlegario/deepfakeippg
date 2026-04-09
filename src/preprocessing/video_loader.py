"""Efficient video loading and frame sampling."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoLoader:
    """Load and sample frames from video files.

    Supports reading video metadata, sampling at a target FPS,
    and handling corrupted or unreadable video files gracefully.
    """

    def __init__(self, target_fps: float = 30.0) -> None:
        """Initialize the video loader.

        Args:
            target_fps: Target frames per second for resampling.
        """
        self.target_fps = target_fps

    def load(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Optional[dict[str, object]]:
        """Load frames from a video file.

        Args:
            video_path: Path to the video file.
            max_frames: Maximum number of frames to load. None for all.

        Returns:
            Dictionary with 'frames' (np.ndarray), 'fps' (float),
            'original_fps' (float), and 'frame_count' (int),
            or None if the video cannot be read.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error("Video not found: %s", video_path)
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return None

        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if original_fps <= 0 or total_frames <= 0:
                logger.error("Invalid video metadata: %s", video_path)
                return None

            # Calculate frame sampling interval
            sample_interval = max(1, int(round(original_fps / self.target_fps)))
            sampled_indices = list(range(0, total_frames, sample_interval))

            if max_frames is not None:
                sampled_indices = sampled_indices[:max_frames]

            frames = []
            frame_idx = 0
            sample_set = set(sampled_indices)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in sample_set:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                frame_idx += 1

                if max_frames is not None and len(frames) >= max_frames:
                    break

            if len(frames) == 0:
                logger.warning("No frames extracted from: %s", video_path)
                return None

            frames_array = np.array(frames, dtype=np.uint8)
            effective_fps = original_fps / sample_interval

            logger.info(
                "Loaded %d frames from %s (%.1f -> %.1f fps)",
                len(frames),
                path.name,
                original_fps,
                effective_fps,
            )

            return {
                "frames": frames_array,
                "fps": effective_fps,
                "original_fps": original_fps,
                "frame_count": len(frames),
                "width": width,
                "height": height,
            }

        except Exception as e:
            logger.error("Error reading video %s: %s", video_path, e)
            return None
        finally:
            cap.release()

    def get_metadata(self, video_path: str) -> Optional[dict[str, float]]:
        """Get video metadata without loading frames.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with fps, frame_count, duration, width, height,
            or None if unreadable.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0

            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "width": width,
                "height": height,
            }
        finally:
            cap.release()
