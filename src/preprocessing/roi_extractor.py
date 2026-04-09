"""Region of Interest (ROI) extraction from detected faces."""

from enum import Enum
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ROIRegion(Enum):
    """Face regions for rPPG signal extraction."""

    FOREHEAD = "R1"       # Forehead region
    LEFT_CHEEK = "R2"     # Left cheek region
    RIGHT_CHEEK = "R3"    # Right cheek region
    ALL = "ALL"           # Combined ROI (entire lower face)


# ROI relative coordinates within the face bounding box.
# Format: (y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio)
ROI_COORDS: dict[ROIRegion, tuple[float, float, float, float]] = {
    ROIRegion.FOREHEAD: (0.10, 0.35, 0.25, 0.75),
    ROIRegion.LEFT_CHEEK: (0.45, 0.75, 0.05, 0.40),
    ROIRegion.RIGHT_CHEEK: (0.45, 0.75, 0.60, 0.95),
    ROIRegion.ALL: (0.10, 0.85, 0.10, 0.90),
}


class ROIExtractor:
    """Extract regions of interest from face bounding boxes.

    Defines standard facial regions (forehead, left cheek, right cheek)
    and extracts mean pixel values for each RGB channel across frames.
    """

    def __init__(
        self,
        regions: Optional[list[ROIRegion]] = None,
    ) -> None:
        """Initialize the ROI extractor.

        Args:
            regions: List of ROI regions to extract. Defaults to all three
                     standard regions (forehead, left cheek, right cheek).
        """
        if regions is None:
            self.regions = [
                ROIRegion.FOREHEAD,
                ROIRegion.LEFT_CHEEK,
                ROIRegion.RIGHT_CHEEK,
            ]
        else:
            self.regions = regions

    def extract_roi(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        region: ROIRegion,
    ) -> Optional[np.ndarray]:
        """Extract a single ROI from a frame given a face bounding box.

        Args:
            frame: RGB image (H, W, 3).
            bbox: Face bounding box (x, y, w, h).
            region: Which facial region to extract.

        Returns:
            ROI pixel patch as numpy array, or None if invalid.
        """
        x, y, w, h = bbox
        coords = ROI_COORDS[region]

        y1 = max(0, y + int(h * coords[0]))
        y2 = min(frame.shape[0], y + int(h * coords[1]))
        x1 = max(0, x + int(w * coords[2]))
        x2 = min(frame.shape[1], x + int(w * coords[3]))

        if y2 <= y1 or x2 <= x1:
            return None

        return frame[y1:y2, x1:x2]

    def extract_mean_rgb(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        region: ROIRegion,
    ) -> Optional[np.ndarray]:
        """Extract mean RGB values from a ROI.

        Args:
            frame: RGB image (H, W, 3).
            bbox: Face bounding box (x, y, w, h).
            region: Which facial region to extract.

        Returns:
            Mean RGB values as array [R, G, B], or None.
        """
        roi = self.extract_roi(frame, bbox, region)
        if roi is None or roi.size == 0:
            return None

        return np.mean(roi.reshape(-1, 3), axis=0).astype(np.float64)

    def extract_all_regions(
        self,
        frames: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict[str, np.ndarray]:
        """Extract mean RGB time series for all configured regions.

        Args:
            frames: Array of RGB frames (N, H, W, 3).
            bbox: Face bounding box (x, y, w, h).

        Returns:
            Dictionary mapping region name to RGB time series (N, 3).
        """
        result: dict[str, np.ndarray] = {}

        for region in self.regions:
            rgb_series = []
            for frame in frames:
                mean_rgb = self.extract_mean_rgb(frame, bbox, region)
                if mean_rgb is not None:
                    rgb_series.append(mean_rgb)
                else:
                    # Use previous value or zeros if first frame
                    if rgb_series:
                        rgb_series.append(rgb_series[-1].copy())
                    else:
                        rgb_series.append(np.zeros(3, dtype=np.float64))

            result[region.value] = np.array(rgb_series, dtype=np.float64)

        logger.info(
            "Extracted %d ROIs with %d frames each",
            len(result),
            len(frames),
        )

        return result
