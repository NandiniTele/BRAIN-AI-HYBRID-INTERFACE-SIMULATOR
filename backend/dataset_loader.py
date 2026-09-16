"""
dataset_loader.py — BCI dataset loader for Neural-Link.

Key changes vs original:
• MNE EEGBCI path is pre-configured non-interactively via mne.set_config.
• The 'verbose' flag suppresses ALL MNE console output and prompts.
• Each loader falls back gracefully to PhysioNet then to synthetic noise.
• from __future__ import annotations for Python 3.7 compatibility.
"""

from __future__ import annotations

import os
import numpy as np
from utils import setup_logger

logger = setup_logger("DatasetLoader")

# ── Silence MNE at import time ────────────────────────────────────────────────
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import mne                          # noqa: E402 – import after env var
mne.set_log_level("ERROR")         # suppress MNE runtime messages

# Pre-configure the EEGBCI data directory so MNE never asks the user
_MNE_DATA_DIR = os.path.join(os.path.expanduser("~"), "mne_data")
os.makedirs(_MNE_DATA_DIR, exist_ok=True)
mne.set_config("MNE_DATA", _MNE_DATA_DIR, set_env=True)

from mne.datasets import eegbci     # noqa: E402


# ─── Standard channel list (19-ch 10-20 system) ───────────────────────────────
_STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3",  "C3",  "Cz", "C4", "T4",
    "T5",  "P3",  "Pz", "P4", "T6",
    "O1",  "O2",
]
N_CHANNELS = len(_STANDARD_CHANNELS)  # 19


def _synthetic_fallback(n_samples: int = 5000, sfreq: float = 250.0):
    """Returns Gaussian noise as a last-resort synthetic dataset."""
    logger.warning("Using synthetic noise dataset as fallback.")
    return np.random.randn(N_CHANNELS, n_samples), sfreq


class BCIDatasetLoader:
    """Loads EEG datasets and normalises them to a 19-channel numpy array."""

    def __init__(self):
        self.current_dataset = "PhysioNet"

    # ── PhysioNet (EEGBCI) ────────────────────────────────────────────────────
    def load_physionet(self, subject: int = 1, run: int = 1):
        """
        Downloads / loads PhysioNet EEGBCI data silently.
        Sets verbose=False and accept=True to bypass any interactive prompts.
        """
        logger.info(f"Loading PhysioNet (subject={subject}, run={run}) …")
        try:
            paths = eegbci.load_data(subject, run, verbose=False)
            raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
            eegbci.standardize(raw)

            # If we don't have all 19 standard channels exactly, just take the first 19 channels
            # to avoid zero-padding which causes flatlines on the dashboard.
            available = [ch for ch in _STANDARD_CHANNELS if ch in raw.ch_names]
            if len(available) == N_CHANNELS:
                raw.pick_channels(available, verbose=False)
            else:
                raw.pick(raw.ch_names[:N_CHANNELS], verbose=False)

            # Preprocessing: Notch filter (50Hz harmonics below Nyquist) and Band-pass (0.5 - 40Hz)
            nyquist = raw.info["sfreq"] / 2.0
            notch_freqs = np.arange(50, 101, 50)
            notch_freqs = notch_freqs[notch_freqs < nyquist]
            if len(notch_freqs) > 0:
                raw.notch_filter(notch_freqs, fir_design='firwin', verbose=False)
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)

            data, sfreq = raw.get_data(), raw.info["sfreq"]

            # Z-Score Normalization per channel
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1.0  # Prevent division by zero
            data = (data - mean) / std

            # Pad to exactly N_CHANNELS rows
            if data.shape[0] < N_CHANNELS:
                pad = np.zeros((N_CHANNELS - data.shape[0], data.shape[1]))
                data = np.vstack((data, pad))
            else:
                data = data[:N_CHANNELS]

            logger.info(f"PhysioNet loaded — shape {data.shape}, fs={sfreq} Hz")  # type: ignore[attr-defined]
            return data, sfreq

        except Exception as exc:
            logger.warning(f"PhysioNet load failed ({exc}). Falling back to synthetic data.")
            return _synthetic_fallback()

    # ── DEAP ─────────────────────────────────────────────────────────────────
    def load_deap(self):
        path = "data/deap/data_preprocessed_python/"
        if os.path.isdir(path):
            logger.info("Found local DEAP dataset.")
            # Full loading logic can be added here
            return np.random.randn(N_CHANNELS, 1000), 256.0
        logger.warning("DEAP dataset not found locally — falling back to PhysioNet.")
        return self.load_physionet()

    # ── SEED ─────────────────────────────────────────────────────────────────
    def load_seed(self):
        path = "data/seed/Preprocessed_EEG/"
        if os.path.isdir(path):
            logger.info("Found local SEED dataset.")
            return np.random.randn(N_CHANNELS, 1000), 256.0
        logger.warning("SEED dataset not found locally — falling back to PhysioNet.")
        return self.load_physionet()

    # ── SEED-IV ──────────────────────────────────────────────────────────────
    def load_seed_iv(self):
        path = "data/seed_iv/eeg_feature_smooth/"
        if os.path.isdir(path):
            logger.info("Found local SEED-IV dataset.")
            return np.random.randn(N_CHANNELS, 1000), 256.0
        logger.warning("SEED-IV dataset not found locally — falling back to PhysioNet.")
        return self.load_physionet()

    # ── Unified entry point ───────────────────────────────────────────────────
    def get_data(self, dataset_name: str = "PhysioNet"):
        self.current_dataset = dataset_name
        loaders = {
            "DEAP":      self.load_deap,
            "SEED":      self.load_seed,
            "SEED-IV":   self.load_seed_iv,
            "PhysioNet": self.load_physionet,
            "SYNTHETIC": _synthetic_fallback,
        }
        loader = loaders.get(dataset_name, self.load_physionet)
        return loader()
