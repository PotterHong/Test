import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, num_speakers, input_dim=40, hidden_dim=128, num_layers=2):
        """
        A simple LSTM-based model for speaker classification.
        Args:
            num_speakers (int): number of unique speakers in the dataset.
            input_dim (int): dimension of the mel-spectrogram feature.
            hidden_dim (int): hidden size for the LSTM.
            num_layers (int): number of LSTM layers.
        """
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, num_speakers)

    def forward(self, x):
        # x: (batch_size, segment_len, input_dim)
        # Pass through LSTM
        out, (h_n, c_n) = self.lstm(x)
        # Take the last time-step's output
        out = out[:, -1, :]  # shape: (batch_size, hidden_dim)
        out = self.classifier(out)
        return out
