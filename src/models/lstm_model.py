"""LSTM model for deepfake detection from rPPG signals."""

import torch
import torch.nn as nn


class DeepfakeLSTM(nn.Module):
    """Bidirectional LSTM for binary classification of rPPG signal sequences.

    Architecture:
        Input (seq_len, n_features) -> LSTM -> Dropout -> FC -> Sigmoid

    The model processes multivariate rPPG time series (9 channels:
    3 ROIs x 3 methods) and outputs a probability of being fake.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output probabilities of shape (batch_size,).
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Use the last hidden state (from both directions if bidirectional)
        last_hidden = lstm_out[:, -1, :]

        # Classification
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Probabilities of shape (batch_size,).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
