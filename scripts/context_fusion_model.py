import torch
import torch.nn as nn

class ContextFusionMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        # Input: [text_prob, audio_prob, sender_reputation, url_risk]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, 4)
        return self.net(x).squeeze(-1)
