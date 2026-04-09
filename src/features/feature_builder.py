"""Feature engineering: build multivariate sequences for model training."""

from typing import Optional

import numpy as np

from src.rppg.green import GreenExtractor
from src.rppg.chrom import ChromExtractor
from src.rppg.pos import POSExtractor
from src.rppg.signal_processor import SignalProcessor
from src.features.segmenter import SignalSegmenter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Build feature sequences from raw ROI RGB time series.

    Extracts GREEN, CHROM, and POS signals from each ROI,
    processes them, and creates multivariate input sequences
    for the LSTM model.

    Output feature vector per timestep (9 channels):
        [ROI1_GREEN, ROI1_CHROM, ROI1_POS,
         ROI2_GREEN, ROI2_CHROM, ROI2_POS,
         ROI3_GREEN, ROI3_CHROM, ROI3_POS]
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        window_seconds: float = 30.0,
        overlap: float = 0.0,
    ) -> None:
        """Initialize the feature builder.

        Args:
            target_fps: Target sampling rate.
            window_seconds: Segment window duration in seconds.
            overlap: Segment overlap fraction.
        """
        self.green = GreenExtractor()
        self.chrom = ChromExtractor()
        self.pos = POSExtractor()
        self.processor = SignalProcessor(target_fps=target_fps)
        self.segmenter = SignalSegmenter(
            window_seconds=window_seconds,
            fps=target_fps,
            overlap=overlap,
        )
        self.target_fps = target_fps

    def extract_signals(
        self,
        roi_rgb: dict[str, np.ndarray],
        original_fps: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Extract and process all rPPG signals from ROI RGB data.

        Args:
            roi_rgb: Dictionary mapping ROI names (R1, R2, R3) to
                     RGB time series of shape (N, 3).
            original_fps: Original video FPS for resampling.

        Returns:
            Dictionary mapping signal names to processed 1D signals.
            Keys are like 'R1_GREEN', 'R1_CHROM', 'R1_POS', etc.
        """
        signals: dict[str, np.ndarray] = {}

        for roi_name, rgb in sorted(roi_rgb.items()):
            # Extract raw signals
            green_raw = self.green.extract(rgb)
            chrom_raw = self.chrom.extract(rgb)
            pos_raw = self.pos.extract(rgb)

            # Process each signal
            for method_name, raw_signal in [
                ("GREEN", green_raw),
                ("CHROM", chrom_raw),
                ("POS", pos_raw),
            ]:
                processed = self.processor.process(raw_signal, original_fps)
                key = f"{roi_name}_{method_name}"
                signals[key] = processed

        logger.info(
            "Extracted %d signals (%d samples each)",
            len(signals),
            len(next(iter(signals.values()))) if signals else 0,
        )

        return signals

    def build_segments(
        self,
        roi_rgb: dict[str, np.ndarray],
        original_fps: Optional[float] = None,
    ) -> list[np.ndarray]:
        """Build multivariate segments from ROI RGB data.

        Args:
            roi_rgb: Dictionary mapping ROI names to RGB time series.
            original_fps: Original video FPS.

        Returns:
            List of segments, each of shape (window_size, 9).
        """
        signals = self.extract_signals(roi_rgb, original_fps)
        segments = self.segmenter.segment_multivariate(signals)
        return segments

    def build_dataset(
        self,
        video_features: list[dict[str, np.ndarray]],
        labels: list[int],
        original_fps_list: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a complete dataset from multiple videos.

        Args:
            video_features: List of ROI RGB dictionaries per video.
            labels: List of labels (0=real, 1=fake) per video.
            original_fps_list: List of original FPS values per video.

        Returns:
            Tuple of (X, y) where X has shape (N_segments, window_size, 9)
            and y has shape (N_segments,).
        """
        all_segments = []
        all_labels = []

        for i, (roi_rgb, label) in enumerate(zip(video_features, labels)):
            fps = original_fps_list[i] if original_fps_list else None
            try:
                segments = self.build_segments(roi_rgb, fps)
                all_segments.extend(segments)
                all_labels.extend([label] * len(segments))
            except Exception as e:
                logger.warning("Failed to process video %d: %s", i, e)
                continue

        if not all_segments:
            logger.error("No segments produced from any video")
            return np.array([]), np.array([])

        x_data = np.array(all_segments, dtype=np.float32)
        y_data = np.array(all_labels, dtype=np.int64)

        logger.info(
            "Built dataset: X=%s, y=%s (real=%d, fake=%d)",
            x_data.shape,
            y_data.shape,
            int(np.sum(y_data == 0)),
            int(np.sum(y_data == 1)),
        )

        return x_data, y_data
