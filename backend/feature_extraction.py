import numpy as np
import scipy.signal as signal
import pywt
from utils import setup_logger

logger = setup_logger("FeatureExtraction")

def extract_features_from_window(data: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    """
    Extracts PSD, Band power, Hjorth, Shannon Entropy, Statistical, and Wavelet features.
    Input: (n_channels, n_samples)
    Output: 1D array of combined features for all channels.
    """
    features = []
    
    for ch_data in data:
        ch_features = []
        
        try:
            # 1. Band Powers (Delta, Theta, Alpha, Beta, Gamma) via Welch's PSD
            nperseg = min(256, len(ch_data))
            freqs, psd = signal.welch(ch_data, sfreq, nperseg=nperseg)
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
            for fmin, fmax in bands.values():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                ch_features.append(np.sum(psd[idx]))
                
            # 2. Hjorth Parameters
            diff1 = np.diff(ch_data)
            diff2 = np.diff(diff1)
            var_y = np.var(ch_data)
            var_d1 = np.var(diff1)
            var_d2 = np.var(diff2)
            
            activity = var_y
            mobility = np.sqrt(var_d1 / var_y) if var_y > 0 else 0
            complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 0 and mobility > 0) else 0
            ch_features.extend([activity, mobility, complexity])
            
            # 3. Shannon Entropy
            p, _ = np.histogram(ch_data, bins=20, density=True)
            p = p[p > 0]
            shannon = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
            ch_features.append(shannon)
            
            # 4. Statistical features
            ch_features.extend([np.mean(ch_data), np.var(ch_data), np.std(ch_data)])
            
            # 5. Wavelet Transform Features
            coeffs = pywt.wavedec(ch_data, 'db4', level=4)
            for c in coeffs:
                ch_features.extend([np.mean(c), np.std(c), np.sum(np.square(c))])
                
        except Exception as e:
            logger.error(f"Error extracting features for a channel: {e}")
            # Pad with zeros if extraction fails
            ch_features = [0.0] * 27
            
        features.extend(ch_features)
        
    return np.array(features)
