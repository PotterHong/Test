# augmentation.py
import torch

def add_random_noise(batch_mels, noise_level=0.01):
    """
    Adds random Gaussian noise to each mel spectrogram in the batch.
    This helps the model become more robust to variations in audio input.
    """
    noise = torch.randn_like(batch_mels) * noise_level
    return batch_mels + noise
