import torch
import random

def random_time_mask(mels, max_masks=2, mask_ratio=0.2):
    """
    Randomly masks a portion of the time dimension for each mel spectrogram.
    """
    processed = []
    for mel in mels:
        length = mel.size(0)
        mask_len = int(length * mask_ratio)
        masked = mel.clone()
        
        for _ in range(max_masks):
            start = random.randint(0, max(0, length - mask_len))
            end = start + mask_len
            masked[start:end, :] = 0.0

        processed.append(masked)
    return torch.stack(processed, dim=0)

def random_freq_mask(mels, max_masks=2, freq_mask_ratio=0.2):
    """
    Randomly masks a portion of the frequency dimension for each mel spectrogram.
    """
    processed = []
    for mel in mels:
        feature_dim = mel.size(1)
        mask_dim = int(feature_dim * freq_mask_ratio)
        masked = mel.clone()
        
        for _ in range(max_masks):
            start = random.randint(0, max(0, feature_dim - mask_dim))
            end = start + mask_dim
            masked[:, start:end] = 0.0
        
        processed.append(masked)
    return torch.stack(processed, dim=0)

def combined_augmentation(batch_mels, 
                          time_mask_ratio=0.2, freq_mask_ratio=0.2,
                          time_mask_count=1, freq_mask_count=1):
    """
    Applies both time masking and frequency masking to the batch.
    """
    batch_mels = random_time_mask(batch_mels, max_masks=time_mask_count, mask_ratio=time_mask_ratio)
    batch_mels = random_freq_mask(batch_mels, max_masks=freq_mask_count, freq_mask_ratio=freq_mask_ratio)
    return batch_mels
