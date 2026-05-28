import torch
import torch.nn as nn
from config import FEATURE_DIM, WINDOW_SIZE


class TemporalAvgClassifier(nn.Module):
    """Simple temporal averaging model for hand-raise detection.

    Takes WINDOW_SIZE consecutive frame features, averages them,
    and passes through a fully connected layer for binary classification.
    """

    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x shape: (batch, window_size, feature_dim)
        x = x.mean(dim=1)  # temporal average -> (batch, feature_dim)
        x = self.fc(x)     # -> (batch, 1)
        return x.squeeze(1)  # -> (batch,)
