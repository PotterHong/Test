import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    """
    A deeper CNN architecture with BatchNorm and Dropout 
    to potentially boost speaker classification performance.
    """
    def __init__(self, num_speakers, input_dim=40, segment_len=128):
        super(EnhancedCNN, self).__init__()
        
        # We increase channels and add more convolutional blocks.
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # No max-pool here; we do an AdaptiveAvgPool below
        )

        # Adjust final pooling to reduce feature map size and handle variable lengths
        self.global_pool = nn.AdaptiveAvgPool2d((segment_len // 8, input_dim // 8))

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128 * (segment_len // 8) * (input_dim // 8), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        # x shape: (batch_size, segment_len, input_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, segment_len, input_dim)
        x = self.conv_block(x) 
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
