"""Face detection using OpenCV's DNN-based detector."""

from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """Detect and track faces in video frames.

    Uses OpenCV's Haar cascade classifier for face detection.
    Detects face on a reference frame and tracks the bounding box
    across subsequent frames.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: tuple[int, int] = (60, 60),
    ) -> None:
        """Initialize the face detector.

        Args:
            scale_factor: Scale factor for the cascade classifier.
            min_neighbors: Minimum neighbors for detection confidence.
            min_face_size: Minimum face size (width, height) in pixels.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Detect the largest face in a single frame.

        Args:
            frame: RGB image as numpy array (H, W, 3).

        Returns:
            Bounding box (x, y, w, h) of the largest face, or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )

        if len(faces) == 0:
            return None

        # Return the largest face by area
        areas = [w * h for (_, _, w, h) in faces]
        largest_idx = int(np.argmax(areas))
        x, y, w, h = faces[largest_idx]

        return (int(x), int(y), int(w), int(h))

    def detect_on_reference(
        self,
        frames: np.ndarray,
        reference_idx: Optional[int] = None,
    ) -> Optional[tuple[int, int, int, int]]:
        """Detect face on a reference frame (mid-frame by default).

        Args:
            frames: Array of frames (N, H, W, 3).
            reference_idx: Index of the reference frame. Defaults to mid-frame.

        Returns:
            Bounding box (x, y, w, h) or None if no face found.
        """
        if reference_idx is None:
            reference_idx = len(frames) // 2

        bbox = self.detect_face(frames[reference_idx])

        if bbox is None:
            # Try a few other frames before giving up
            for offset in [0, len(frames) // 4, 3 * len(frames) // 4]:
                idx = min(offset, len(frames) - 1)
                bbox = self.detect_face(frames[idx])
                if bbox is not None:
                    logger.info(
                        "Face found on fallback frame %d (ref frame %d failed)",
                        idx,
                        reference_idx,
                    )
                    break

        if bbox is None:
            logger.warning("No face detected in any reference frame")

        return bbox

    def get_stable_bbox(
        self,
        frames: np.ndarray,
        num_samples: int = 5,
    ) -> Optional[tuple[int, int, int, int]]:
        """Get a stable bounding box by averaging detections across frames.

        Args:
            frames: Array of frames (N, H, W, 3).
            num_samples: Number of frames to sample for averaging.

        Returns:
            Averaged bounding box (x, y, w, h) or None.
        """
        n_frames = len(frames)
        sample_indices = np.linspace(0, n_frames - 1, num_samples, dtype=int)

        bboxes = []
        for idx in sample_indices:
            bbox = self.detect_face(frames[idx])
            if bbox is not None:
                bboxes.append(bbox)

        if len(bboxes) == 0:
            logger.warning("No face detected in any sampled frame")
            return None

        # Average the bounding boxes
        avg_bbox = np.mean(bboxes, axis=0).astype(int)
        return (int(avg_bbox[0]), int(avg_bbox[1]), int(avg_bbox[2]), int(avg_bbox[3]))
