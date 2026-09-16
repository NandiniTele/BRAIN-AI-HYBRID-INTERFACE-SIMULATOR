import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess_window
from feature_extraction import extract_features_from_window
from model import HybridBCINet

def test_preprocessing():
    # Mock EEG data: 19 channels, 250 samples
    data = np.random.randn(19, 250)
    preprocessed = preprocess_window(data, sfreq=250.0)
    assert preprocessed.shape == (19, 250)
    # Check baseline correction (mean roughly 0)
    assert np.allclose(np.mean(preprocessed, axis=1), 0, atol=1e-1)

def test_feature_extraction():
    data = np.random.randn(19, 250)
    features = extract_features_from_window(data, sfreq=250.0)
    # 19 channels * 27 features = 513
    assert features.shape == (513,)

def test_model_forward():
    model = HybridBCINet(input_features=513, num_channels=19)
    # Batch size 1, 19 channels, 27 features per channel
    mock_input = torch.randn(1, 19, 27)
    
    with torch.no_grad():
        outputs = model(mock_input)
        
    assert "emotion_logits" in outputs
    assert outputs["emotion_logits"].shape == (1, 5)
    assert 0 <= outputs["focus"].item() <= 1
