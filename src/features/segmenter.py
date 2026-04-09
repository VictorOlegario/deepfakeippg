"""Signal segmentation into fixed-length windows."""

from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalSegmenter:
    """Segment rPPG signals into fixed-length windows.

    Creates overlapping or non-overlapping segments from
    continuous signals for model input.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        fps: float = 30.0,
        overlap: float = 0.0,
        min_snr: Optional[float] = None,
    ) -> None:
        """Initialize the segmenter.

        Args:
            window_seconds: Window duration in seconds.
            fps: Signal sampling rate in Hz.
            overlap: Fraction of window overlap (0.0 to 0.9).
            min_snr: Minimum SNR threshold to keep a segment. None to keep all.
        """
        self.window_size = int(window_seconds * fps)
        self.fps = fps
        self.overlap = overlap
        self.min_snr = min_snr
        self.step_size = int(self.window_size * (1.0 - overlap))

        if self.step_size < 1:
            self.step_size = 1

    def segment(self, signal: np.ndarray) -> list[np.ndarray]:
        """Segment a 1D signal into windows.

        Args:
            signal: Input signal (N,).

        Returns:
            List of signal segments, each of shape (window_size,).
        """
        segments = []
        n = len(signal)

        if n < self.window_size:
            logger.warning(
                "Signal length (%d) < window size (%d). Padding.",
                n,
                self.window_size,
            )
            padded = np.zeros(self.window_size, dtype=signal.dtype)
            padded[:n] = signal
            return [padded]

        start = 0
        while start + self.window_size <= n:
            segment = signal[start:start + self.window_size]
            segments.append(segment)
            start += self.step_size

        logger.debug(
            "Created %d segments (window=%d, step=%d) from %d samples",
            len(segments),
            self.window_size,
            self.step_size,
            n,
        )

        return segments

    def segment_multivariate(
        self,
        signals: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Segment multiple aligned signals into multivariate windows.

        Args:
            signals: Dictionary mapping signal names to 1D arrays.
                     All signals must have the same length.

        Returns:
            List of multivariate segments, each of shape
            (window_size, n_channels).
        """
        # Stack all signals into a matrix
        names = sorted(signals.keys())
        arrays = [signals[name] for name in names]

        # Verify all same length
        lengths = [len(a) for a in arrays]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            logger.warning(
                "Signal length mismatch (%s). Truncating to %d.",
                lengths,
                min_len,
            )
            arrays = [a[:min_len] for a in arrays]

        # Shape: (N, n_channels)
        stacked = np.column_stack(arrays)
        n = stacked.shape[0]

        segments = []
        start = 0
        while start + self.window_size <= n:
            segment = stacked[start:start + self.window_size]
            segments.append(segment)
            start += self.step_size

        if len(segments) == 0 and n > 0:
            padded = np.zeros(
                (self.window_size, stacked.shape[1]), dtype=stacked.dtype
            )
            padded[:n] = stacked
            segments.append(padded)

        logger.debug(
            "Created %d multivariate segments (%d channels)",
            len(segments),
            stacked.shape[1],
        )

        return segments
