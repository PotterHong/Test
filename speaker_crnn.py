# import torch
# import torch.nn as nn

# class CRNN(nn.Module):
    
#     def __init__(self, num_speakers, input_dim=40, segment_len=128, hidden_dim=128, rnn_layers=2):
#         super(CRNN, self).__init__()
    
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2)),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )

#         self.rnn = nn.LSTM(
#             input_size=(input_dim // 4) * 64,
#             hidden_size=hidden_dim,
#             num_layers=rnn_layers,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_speakers)
#         )
        
#     def forward(self, x):
#         # x: (B, T, F)
#         x = x.unsqueeze(1)
#         x = self.cnn(x)
#         x = x.permute(0, 2, 1, 3).contiguous()
#         x = x.view(x.size(0), x.size(1), -1)
#         x, _ = self.rnn(x)
#         x = x[:, -1, :]
#         x = self.fc(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CRNN(nn.Module):
    def __init__(self, num_speakers, input_dim=40, segment_len=128, hidden_dim=128, rnn_layers=2):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.rnn = nn.LSTM(
            input_size=(input_dim // 4) * 64,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attn = nn.Linear(hidden_dim * 2, 1)  # Self-attention over time

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_speakers)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.cnn(x)     # (B, C, T', F')
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T', C, F')
        x = x.view(x.size(0), x.size(1), -1)    # (B, T', C*F')

        rnn_out, _ = self.rnn(x)                # (B, T', 2*hidden_dim)
        rnn_out = self.layer_norm(rnn_out)

        # Attention pooling over time
        attn_weights = F.softmax(self.attn(rnn_out), dim=1)  # (B, T', 1)
        x = torch.sum(rnn_out * attn_weights, dim=1)         # (B, 2*hidden_dim)

        x = self.fc(x)
        return x
