# speaker_cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_speakers, input_dim=40, segment_len=128):
        """
        A basic convolutional neural network for speaker classification.
        Args:
            num_speakers (int): number of unique speakers in the dataset.
            input_dim (int): dimension of the mel-spectrogram feature (e.g., 40).
            segment_len (int): length of the mel-spectrogram segments (e.g., 128).
        """
        super(SimpleCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # After two max-pool ops, each dimension is roughly reduced by factor of 4.
        # So shape is ~ (batch_size, 32, segment_len//4, input_dim//4).
        self.fc = nn.Sequential(
            nn.Linear(32 * (segment_len // 4) * (input_dim // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_speakers)
        )

    def forward(self, x):
        # x shape: (batch_size, segment_len, input_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, segment_len, input_dim)
        out = self.conv(x)  # -> shape is (batch_size, 32, segment_len//4, input_dim//4)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out
