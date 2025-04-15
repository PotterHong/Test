import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    A basic residual block for our naive ResNet
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)  # Add the skip connection
        out = self.relu(out)
        
        return out

class NaiveResNet(nn.Module):
    def __init__(self, num_speakers, input_dim=40, segment_len=128, block=BasicBlock):
        """
        A naive ResNet architecture for speaker classification.
        
        Args:
            num_speakers (int): number of unique speakers in the dataset.
            input_dim (int): dimension of the mel-spectrogram feature (e.g., 40).
            segment_len (int): length of the mel-spectrogram segments (e.g., 128).
            block (nn.Module): the block to use (default: BasicBlock)
        """
        super(NaiveResNet, self).__init__()
        
        self.in_channels = 16
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 16, 2, stride=1)
        self.layer2 = self._make_layer(block, 32, 2, stride=2)
        self.layer3 = self._make_layer(block, 64, 2, stride=2)
        
        # Calculate output size after convolutions
        conv_output_height = segment_len // 4  # After 2 stride-2 operations
        conv_output_width = input_dim // 4     # After 2 stride-2 operations
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(64, num_speakers)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, segment_len, input_dim)
        x = x.unsqueeze(1)  # -> (batch_size, 1, segment_len, input_dim)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        
        return out