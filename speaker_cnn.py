# speaker_cnn.py
import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_speakers, input_dim=40):
        """
        A basic convolutional neural network for speaker classification.
        Args:
            num_speakers (int): number of unique speakers in the dataset.
            input_dim (int): dimension of the mel-spectrogram feature.
        """
        super(SimpleCNN, self).__init__()
        
        # We'll reshape input to (batch_size, 1, mel_length, mel_dim) for convolution.
        # Then apply a couple of convolution + pooling layers, and a linear layer for classification.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # After two pooling layers, the mel_length is reduced by a factor of 4,
        # and the mel_dim is also reduced by a factor of 4 (assuming each dimension is at least that large).
        # We'll flatten and feed it to a linear layer.
        self.fc = nn.Sequential(
            nn.Linear(32 * (input_dim // 4) * (128 // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_speakers)
        )

    def forward(self, x):
        # x shape is (batch_size, mel_length, mel_dim).
        # Reshape for convolution: (batch_size, 1, mel_length, mel_dim).
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
