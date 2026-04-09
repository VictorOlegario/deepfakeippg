"""GREEN channel rPPG signal extraction.

The simplest rPPG method: uses the mean green channel intensity
as a proxy for the blood volume pulse signal.

Reference:
    Verkruysse, W., et al. (2008). Remote plethysmographic imaging using
    ambient light. Optics Express, 16(26), 21434-21445.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GreenExtractor:
    """Extract rPPG signal using the GREEN channel method.

    The green channel of skin pixels contains the strongest
    plethysmographic signal due to hemoglobin absorption.
    """

    def extract(self, rgb_signal: np.ndarray) -> np.ndarray:
        """Extract GREEN rPPG signal from mean RGB time series.

        Args:
            rgb_signal: Mean RGB values per frame, shape (N, 3),
                        columns are [R, G, B].

        Returns:
            Green channel signal, shape (N,).
        """
        if rgb_signal.ndim != 2 or rgb_signal.shape[1] != 3:
            raise ValueError(
                f"Expected shape (N, 3), got {rgb_signal.shape}"
            )

        green_signal = rgb_signal[:, 1].copy()
        logger.debug("GREEN signal extracted: %d samples", len(green_signal))

        return green_signal
