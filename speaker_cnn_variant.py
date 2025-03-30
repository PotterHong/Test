import torch
import torch.nn as nn

class AnotherCNN(nn.Module):
    def __init__(self, num_speakers, input_dim=40, segment_len=128):
        """
        A slightly deeper CNN for speaker classification.
        Args:
            num_speakers (int): number of unique speakers in the dataset.
            input_dim (int): dimension of the mel-spectrogram feature.
            segment_len (int): length of the mel-spectrogram segments.
        """
        super(AnotherCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Adaptive pooling to handle varied input sizes in a more robust way:
            nn.AdaptiveAvgPool2d((segment_len // 8, input_dim // 8))
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (segment_len // 8) * (input_dim // 8), 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        # x shape: (batch_size, segment_len, input_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, segment_len, input_dim)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
