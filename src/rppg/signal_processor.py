"""Signal processing utilities for rPPG signals.

Implements resampling, detrending, normalization, and bandpass filtering.
"""

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalProcessor:
    """Process raw rPPG signals for analysis and feature extraction.

    Applies a standard pipeline:
    1. Resample to fixed FPS
    2. Detrend (remove slow drift)
    3. Normalize (zero mean, unit variance)
    4. Bandpass filter (0.7-3.0 Hz = 42-180 BPM)
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        lowcut: float = 0.7,
        highcut: float = 3.0,
        filter_order: int = 4,
    ) -> None:
        """Initialize signal processor.

        Args:
            target_fps: Target sampling rate in Hz.
            lowcut: Lower cutoff frequency in Hz (0.7 Hz = 42 BPM).
            highcut: Upper cutoff frequency in Hz (3.0 Hz = 180 BPM).
            filter_order: Butterworth filter order.
        """
        self.target_fps = target_fps
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order

    def resample(
        self,
        sig: np.ndarray,
        original_fps: float,
    ) -> np.ndarray:
        """Resample signal to target FPS.

        Args:
            sig: Input signal (N,).
            original_fps: Original sampling rate.

        Returns:
            Resampled signal.
        """
        if abs(original_fps - self.target_fps) < 0.1:
            return sig.copy()

        n_original = len(sig)
        duration = n_original / original_fps
        n_target = int(duration * self.target_fps)

        t_original = np.linspace(0, duration, n_original, endpoint=False)
        t_target = np.linspace(0, duration, n_target, endpoint=False)

        interpolator = interp1d(
            t_original, sig, kind="cubic", fill_value="extrapolate"
        )
        resampled = interpolator(t_target)

        return resampled.astype(np.float64)

    def detrend(self, sig: np.ndarray, lambda_val: float = 100.0) -> np.ndarray:
        """Remove slow drift from signal using scipy detrend.

        Args:
            sig: Input signal (N,).
            lambda_val: Regularization parameter (unused, kept for API).

        Returns:
            Detrended signal.
        """
        return scipy_signal.detrend(sig, type="linear").astype(np.float64)

    def normalize(self, sig: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance.

        Args:
            sig: Input signal (N,).

        Returns:
            Normalized signal.
        """
        std = np.std(sig)
        if std < 1e-10:
            return sig - np.mean(sig)
        return ((sig - np.mean(sig)) / std).astype(np.float64)

    def bandpass_filter(self, sig: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
        """Apply Butterworth bandpass filter.

        Args:
            sig: Input signal (N,).
            fs: Sampling frequency. Defaults to target_fps.

        Returns:
            Filtered signal.
        """
        if fs is None:
            fs = self.target_fps

        nyquist = fs / 2.0

        if self.highcut >= nyquist:
            logger.warning(
                "Highcut (%.1f Hz) >= Nyquist (%.1f Hz). Adjusting.",
                self.highcut,
                nyquist,
            )
            highcut = nyquist * 0.95
        else:
            highcut = self.highcut

        if self.lowcut <= 0:
            lowcut = 0.01
        else:
            lowcut = self.lowcut

        sos = scipy_signal.butter(
            self.filter_order,
            [lowcut / nyquist, highcut / nyquist],
            btype="band",
            output="sos",
        )

        filtered = scipy_signal.sosfiltfilt(sos, sig).astype(np.float64)
        return filtered

    def process(
        self,
        sig: np.ndarray,
        original_fps: Optional[float] = None,
    ) -> np.ndarray:
        """Apply the full processing pipeline.

        Args:
            sig: Raw rPPG signal (N,).
            original_fps: Original sampling rate. If None, skip resampling.

        Returns:
            Processed signal.
        """
        processed = sig.copy()

        # Step 1: Resample
        if original_fps is not None:
            processed = self.resample(processed, original_fps)

        # Step 2: Detrend
        processed = self.detrend(processed)

        # Step 3: Normalize
        processed = self.normalize(processed)

        # Step 4: Bandpass filter
        if len(processed) > 3 * self.filter_order:
            processed = self.bandpass_filter(processed)
        else:
            logger.warning(
                "Signal too short (%d samples) for filtering", len(processed)
            )

        return processed

    def compute_snr(self, sig: np.ndarray, fs: Optional[float] = None) -> float:
        """Compute signal-to-noise ratio using power spectral density.

        Args:
            sig: Processed signal (N,).
            fs: Sampling frequency. Defaults to target_fps.

        Returns:
            SNR in dB.
        """
        if fs is None:
            fs = self.target_fps

        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))

        # Signal band: 0.7-3.0 Hz
        signal_mask = (freqs >= self.lowcut) & (freqs <= self.highcut)
        noise_mask = ~signal_mask & (freqs > 0)

        signal_power = np.sum(psd[signal_mask])
        noise_power = np.sum(psd[noise_mask])

        if noise_power < 1e-10:
            return float("inf")

        snr = 10.0 * np.log10(signal_power / noise_power)
        return float(snr)

    def estimate_bpm(self, sig: np.ndarray, fs: Optional[float] = None) -> float:
        """Estimate heart rate (BPM) from signal using FFT.

        Args:
            sig: Processed signal (N,).
            fs: Sampling frequency. Defaults to target_fps.

        Returns:
            Estimated heart rate in BPM.
        """
        if fs is None:
            fs = self.target_fps

        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))

        # Only consider frequencies in the valid HR range
        valid_mask = (freqs >= self.lowcut) & (freqs <= self.highcut)
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]

        if len(valid_psd) == 0:
            return 0.0

        peak_freq = valid_freqs[np.argmax(valid_psd)]
        bpm = peak_freq * 60.0

        return float(bpm)
