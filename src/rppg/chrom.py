"""CHROM (Chrominance-based) rPPG signal extraction.

Reference:
    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from
    chrominance-based rPPG. IEEE Transactions on Biomedical
    Engineering, 60(10), 2878-2886.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromExtractor:
    """Extract rPPG signal using the CHROM method.

    Projects normalized RGB signals onto a chrominance plane
    to separate the pulse signal from motion artifacts.
    """

    def __init__(self, window_size: int = 45) -> None:
        """Initialize CHROM extractor.

        Args:
            window_size: Sliding window size in frames for local normalization.
        """
        self.window_size = window_size

    def extract(self, rgb_signal: np.ndarray) -> np.ndarray:
        """Extract CHROM rPPG signal from mean RGB time series.

        Args:
            rgb_signal: Mean RGB values per frame, shape (N, 3),
                        columns are [R, G, B].

        Returns:
            CHROM signal, shape (N,).
        """
        if rgb_signal.ndim != 2 or rgb_signal.shape[1] != 3:
            raise ValueError(
                f"Expected shape (N, 3), got {rgb_signal.shape}"
            )

        n_frames = rgb_signal.shape[0]
        chrom_signal = np.zeros(n_frames, dtype=np.float64)

        for start in range(0, n_frames, self.window_size):
            end = min(start + self.window_size, n_frames)
            window = rgb_signal[start:end].copy()

            if len(window) < 3:
                continue

            # Normalize each channel by its mean
            means = np.mean(window, axis=0)
            means = np.where(means == 0, 1.0, means)
            normalized = window / means

            r_n = normalized[:, 0]
            g_n = normalized[:, 1]
            b_n = normalized[:, 2]

            # Chrominance signals
            x_s = 3.0 * r_n - 2.0 * g_n
            y_s = 1.5 * r_n + g_n - 1.5 * b_n

            # Combine with adaptive weighting
            std_xs = np.std(x_s)
            std_ys = np.std(y_s)

            if std_ys > 0:
                alpha = std_xs / std_ys
            else:
                alpha = 0.0

            chrom_window = x_s - alpha * y_s

            # Overlap-add
            chrom_signal[start:end] += chrom_window

        logger.debug("CHROM signal extracted: %d samples", n_frames)
        return chrom_signal
