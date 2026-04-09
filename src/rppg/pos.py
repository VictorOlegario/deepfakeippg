"""POS (Plane Orthogonal to Skin) rPPG signal extraction.

Reference:
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic principles of remote PPG. IEEE Transactions on
    Biomedical Engineering, 64(7), 1479-1491.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class POSExtractor:
    """Extract rPPG signal using the POS method.

    Projects the temporal RGB signal onto a plane orthogonal
    to the skin-tone direction, suppressing motion-induced
    color variations.
    """

    def __init__(self, window_size: int = 45) -> None:
        """Initialize POS extractor.

        Args:
            window_size: Sliding window size in frames.
        """
        self.window_size = window_size

    def extract(self, rgb_signal: np.ndarray) -> np.ndarray:
        """Extract POS rPPG signal from mean RGB time series.

        Args:
            rgb_signal: Mean RGB values per frame, shape (N, 3),
                        columns are [R, G, B].

        Returns:
            POS signal, shape (N,).
        """
        if rgb_signal.ndim != 2 or rgb_signal.shape[1] != 3:
            raise ValueError(
                f"Expected shape (N, 3), got {rgb_signal.shape}"
            )

        n_frames = rgb_signal.shape[0]
        pos_signal = np.zeros(n_frames, dtype=np.float64)
        counts = np.zeros(n_frames, dtype=np.float64)

        for start in range(0, n_frames - self.window_size + 1):
            end = start + self.window_size
            window = rgb_signal[start:end].copy()

            # Temporal normalization
            means = np.mean(window, axis=0)
            means = np.where(means == 0, 1.0, means)
            c_n = window / means

            # POS projection
            # S1 = C_n_G - C_n_B
            # S2 = C_n_G + C_n_B - 2 * C_n_R
            s1 = c_n[:, 1] - c_n[:, 2]
            s2 = c_n[:, 1] + c_n[:, 2] - 2.0 * c_n[:, 0]

            # Adaptive combination
            std_s1 = np.std(s1)
            std_s2 = np.std(s2)

            if std_s2 > 0:
                alpha = std_s1 / std_s2
            else:
                alpha = 0.0

            h = s1 + alpha * s2

            # Overlap-add
            pos_signal[start:end] += h
            counts[start:end] += 1.0

        # Normalize by overlap count
        counts = np.where(counts == 0, 1.0, counts)
        pos_signal = pos_signal / counts

        logger.debug("POS signal extracted: %d samples", n_frames)
        return pos_signal
