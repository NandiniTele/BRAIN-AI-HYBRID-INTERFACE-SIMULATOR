"""
preprocessing.py — EEG signal preprocessing pipeline for Neural-Link.

Changes vs original:
• All MNE calls use verbose=False to suppress console output.
• ICA convergence warnings are caught silently.
• from __future__ import annotations for Python 3.7 compatibility.
"""

from __future__ import annotations

import warnings
import numpy as np
import mne
from utils import setup_logger

logger = setup_logger("Preprocessing")

# Suppress sklearn/MNE convergence warnings that are expected for short windows
warnings.filterwarnings("ignore", message="FastICA did not converge")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def preprocess_window(
    data: np.ndarray,
    sfreq: float = 250.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """
    Applies standard BCI preprocessing to a (channels × samples) array:
      1. Band-pass filter  (0.5 – 45 Hz)
      2. Notch filter      (50 Hz powerline)
      3. ICA artifact removal  (optional, skipped silently on error)
      4. Baseline correction   (subtract channel mean)
      5. Z-score normalisation
    """
    try:
        # 1. Band-pass
        data = mne.filter.filter_data(
            data, sfreq, l_freq, h_freq,
            method="fir", fir_window="hamming",
            verbose=False,
        )

        # 2. Notch filter (dynamic below Nyquist)
        nyquist = sfreq / 2.0
        # Create harmonics of the notch frequency
        notch_freqs = np.arange(notch_freq, nyquist, notch_freq)
        if len(notch_freqs) > 0:
            data = mne.filter.notch_filter(
                data, sfreq, freqs=notch_freqs,
                method="fir",
                verbose=False,
            )

        # 3. ICA (optional)
        try:
            from sklearn.decomposition import FastICA
            # Avoid ICA on very short windows
            if data.shape[1] > sfreq:
                ica = FastICA(n_components=data.shape[0], random_state=42, max_iter=200, tol=0.01)
                data = ica.fit_transform(data.T).T
        except Exception as e:
            pass

        # 4. Bad Channel Detection and Mean Interpolation
        # Variance-based outlier detection (> 3 sigma from median variance)
        variances = np.var(data, axis=1)
        median_var = np.median(variances)
        mad = np.median(np.abs(variances - median_var))
        if mad > 0:
            z_scores = 0.6745 * (variances - median_var) / mad
            bad_channels = np.where(np.abs(z_scores) > 3.5)[0]
            if len(bad_channels) > 0 and len(bad_channels) < data.shape[0] - 1:
                good_channels = np.setdiff1d(np.arange(data.shape[0]), bad_channels)
                mean_good = np.mean(data[good_channels], axis=0)
                for bc in bad_channels:
                    data[bc] = mean_good

        # 5. Baseline correction
        data = data - np.mean(data, axis=1, keepdims=True)

        # 5. Z-Score Normalization
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1.0
        data = (data - mean) / std

        return data

    except Exception as exc:
        logger.error(f"Preprocessing error: {exc} — returning raw data.")
        return data
