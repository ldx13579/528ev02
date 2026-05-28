import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from config import FEATURE_DIM, WINDOW_SIZE, TCN_NUM_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT


class TemporalAvgClassifier(nn.Module):
    """Baseline: temporal averaging model for comparison."""

    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x shape: (batch, window_size, feature_dim)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x.squeeze(1)


class CausalConv1d(nn.Module):
    """Causal convolution with left-side padding only."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual block with two dilated causal convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu_out(out + residual)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network for hand-raise detection.

    Input: (batch, window_size=30, feature_dim=1280)
    Uses stacked dilated causal convolution blocks to capture
    multi-scale temporal patterns.
    """

    def __init__(self, feature_dim=FEATURE_DIM, num_channels=None,
                 kernel_size=TCN_KERNEL_SIZE, dropout=TCN_DROPOUT):
        super().__init__()
        if num_channels is None:
            num_channels = TCN_NUM_CHANNELS

        self.input_proj = nn.Linear(feature_dim, num_channels[0])

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_channels[0] if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)  # (batch, seq_len, num_channels[0])
        x = x.transpose(1, 2)  # (batch, channels, seq_len) for Conv1d
        x = self.tcn(x)        # (batch, channels[-1], seq_len)
        x = x[:, :, -1]        # take last timestep output (causal)
        x = self.classifier(x) # (batch, 1)
        return x.squeeze(1)
