import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """A basic ResNet block with skip connections.
    
    This block consists of two 3x3 convolutions with batch normalization,
    ReLU activations, and a skip connection.
    
    Attributes:
        expansion (int): Channel expansion factor (1 for BasicBlock)
        conv1, conv2: Convolutional layers
        bn1, bn2: Batch normalization layers
        shortcut: Skip connection path
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        """Initialize the BasicBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for first conv layer. Defaults to 1.
        """
        super(BasicBlock, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass through the BasicBlock.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output after processing through the block
        """
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection and final activation
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions.
    
    This block uses a bottleneck architecture to reduce computation:
    1. 1x1 conv to reduce channels
    2. 3x3 conv for spatial processing
    3. 1x1 conv to increase channels
    
    Attributes:
        expansion (int): Channel expansion factor (4 for Bottleneck)
        conv1, conv2, conv3: Convolutional layers
        bn1, bn2, bn3: Batch normalization layers
        shortcut: Skip connection path
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        """Initialize the Bottleneck block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of intermediate channels
            stride (int, optional): Stride for the 3x3 conv. Defaults to 1.
        """
        super(Bottleneck, self).__init__()
        
        # First 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv for spatial processing
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Second 1x1 conv to increase channels
        self.conv3 = nn.Conv2d(
            out_channels, 
            out_channels * self.expansion, 
            kernel_size=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        """Forward pass through the Bottleneck block.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output after processing through the block
        """
        identity = x
        
        # First 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Second 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection and final activation
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class SpeakerResNet(nn.Module):
    """ResNet architecture for speaker verification and spoofing detection.
    
    This model processes mel-spectrogram features to:
    1. Identify speakers (classification)
    2. Detect synthetic/spoofed speech
    3. Extract speaker embeddings
    
    The architecture includes attention mechanisms to focus on
    relevant time-frequency regions.
    
    Attributes:
        in_channels (int): Current channel dimension
        conv1: Initial convolution layer
        bn1: Initial batch normalization
        layer1-4: ResNet block layers
        avg_pool: Global average pooling
        fc: Speaker classification layer
        spoof_detector: Binary classifier for spoofing detection
        attention: Channel attention mechanism
    """
    
    def __init__(self, block, num_blocks, num_speakers, input_dim=40, segment_len=128):
        """Initialize the SpeakerResNet model.
        
        Args:
            block (nn.Module): Block type (BasicBlock or Bottleneck)
            num_blocks (list): Number of blocks in each layer
            num_speakers (int): Number of speakers to classify
            input_dim (int, optional): Frequency dimension. Defaults to 40.
            segment_len (int, optional): Time dimension. Defaults to 128.
        """
        super(SpeakerResNet, self).__init__()
        self.in_channels = 64
        
        # Initial processing
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Output dimensions
        self.final_channels = 512 * block.expansion
        
        # Global average pooling and speaker classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.final_channels, num_speakers)
        
        # Spoofing detection classifier
        self.spoof_detector = nn.Sequential(
            nn.Linear(self.final_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(self.final_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.final_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple blocks.
        
        Args:
            block (nn.Module): Block type (BasicBlock or Bottleneck)
            out_channels (int): Number of output channels
            num_blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: A sequence of ResNet blocks
        """
        # First block may downsample with stride>1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the SpeakerResNet.
        
        Args:
            x (Tensor): Input mel-spectrogram [batch_size, segment_len, input_dim]
            
        Returns:
            tuple: (speaker_logits, spoof_probability, embedding)
        """
        batch_size = x.size(0)
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, segment_len, input_dim]
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention mechanism
        attn = self.attention(x)
        x = x * attn
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, channels]
        
        # Get embeddings (features before classification)
        embedding = x
        
        # Speaker classification
        speaker_logits = self.fc(embedding)
        
        # Spoofing detection
        spoof_prob = self.spoof_detector(embedding)
        
        return speaker_logits, spoof_prob, embedding
    
    def get_embedding(self, x):
        """Extract speaker embedding from an input mel-spectrogram.
        
        This is a utility method for speaker verification tasks.
        
        Args:
            x (Tensor): Input mel-spectrogram [batch_size, segment_len, input_dim]
            
        Returns:
            Tensor: Speaker embedding vectors [batch_size, embedding_dim]
        """
        batch_size = x.size(0)
        
        # Process through the network
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        attn = self.attention(x)
        x = x * attn
        
        # Global pooling and flatten
        x = self.avg_pool(x)
        embedding = x.view(batch_size, -1)
        
        return embedding


# Factory functions to create different ResNet variants
def ResNet18(num_speakers, input_dim=40, segment_len=128):
    """Create a ResNet-18 model for speaker verification.
    
    Args:
        num_speakers (int): Number of speakers to classify
        input_dim (int, optional): Frequency dimension. Defaults to 40.
        segment_len (int, optional): Time dimension. Defaults to 128.
        
    Returns:
        SpeakerResNet: A ResNet-18 model
    """
    return SpeakerResNet(BasicBlock, [2, 2, 2, 2], num_speakers, input_dim, segment_len)


def ResNet34(num_speakers, input_dim=40, segment_len=128):
    """Create a ResNet-34 model for speaker verification.
    
    Args:
        num_speakers (int): Number of speakers to classify
        input_dim (int, optional): Frequency dimension. Defaults to 40.
        segment_len (int, optional): Time dimension. Defaults to 128.
        
    Returns:
        SpeakerResNet: A ResNet-34 model
    """
    return SpeakerResNet(BasicBlock, [3, 4, 6, 3], num_speakers, input_dim, segment_len)


def ResNet50(num_speakers, input_dim=40, segment_len=128):
    """Create a ResNet-50 model for speaker verification.
    
    Args:
        num_speakers (int): Number of speakers to classify
        input_dim (int, optional): Frequency dimension. Defaults to 40.
        segment_len (int, optional): Time dimension. Defaults to 128.
        
    Returns:
        SpeakerResNet: A ResNet-50 model
    """
    return SpeakerResNet(Bottleneck, [3, 4, 6, 3], num_speakers, input_dim, segment_len)


def ResNet101(num_speakers, input_dim=40, segment_len=128):
    """Create a ResNet-101 model for speaker verification.
    
    Args:
        num_speakers (int): Number of speakers to classify
        input_dim (int, optional): Frequency dimension. Defaults to 40.
        segment_len (int, optional): Time dimension. Defaults to 128.
        
    Returns:
        SpeakerResNet: A ResNet-101 model
    """
    return SpeakerResNet(Bottleneck, [3, 4, 23, 3], num_speakers, input_dim, segment_len)