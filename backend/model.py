import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super(ResidualCNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout1d(p=dropout_p)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout1d(p=dropout_p)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.bn2(self.conv2(out))
        out = self.drop2(out)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class HybridBCINet(nn.Module):
    def __init__(self, input_features: int, num_channels: int = 19, 
                 cnn_filters: int = 128, lstm_units: int = 128, 
                 num_heads: int = 8, dropout_p: float = 0.3):
        super(HybridBCINet, self).__init__()
        
        # Residual CNN Feature Extractor
        self.res1 = ResidualCNNBlock(num_channels, cnn_filters // 2, dropout_p=dropout_p)
        self.res2 = ResidualCNNBlock(cnn_filters // 2, cnn_filters, dropout_p=dropout_p)
        self.res3 = ResidualCNNBlock(cnn_filters, cnn_filters, dropout_p=dropout_p)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_p)
        
        # Multi-Head Self Attention & Transformer Encoder
        embed_dim = lstm_units * 2 # Bidirectional
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_p)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=dropout_p, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.ln = nn.LayerNorm(embed_dim)
        
        # Global Average Pooling (GAP) is done in the forward pass implicitly via mean(dim=1)
        
        # Dense classifier heads (5 emotion classes)
        self.fc_emotion = nn.Linear(embed_dim, 5)
        self.fc_focus = nn.Linear(embed_dim, 1)
        self.fc_attention = nn.Linear(embed_dim, 1)
        self.fc_stress = nn.Linear(embed_dim, 1)
        self.fc_fatigue = nn.Linear(embed_dim, 1)
        self.fc_cog_load = nn.Linear(embed_dim, 1)
        self.fc_valence = nn.Linear(embed_dim, 1)
        self.fc_arousal = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x: (batch, num_channels, features_per_channel)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = x.transpose(1, 2) # (batch, seq, cnn_filters)
        
        lstm_out, _ = self.lstm(x) # (batch, seq, lstm_units*2)
        
        # Multi-Head Self Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(attn_out) 
        
        # Global Average Pooling across the temporal dimension
        out = transformer_out.mean(dim=1) 
        
        out = self.ln(out)
        
        return {
            "emotion_logits": self.fc_emotion(out),
            "focus": torch.sigmoid(self.fc_focus(out)),
            "attention": torch.sigmoid(self.fc_attention(out)),
            "stress": torch.sigmoid(self.fc_stress(out)),
            "fatigue": torch.sigmoid(self.fc_fatigue(out)),
            "cognitive_load": torch.sigmoid(self.fc_cog_load(out)),
            "valence": torch.sigmoid(self.fc_valence(out)),
            "arousal": torch.sigmoid(self.fc_arousal(out))
        }
