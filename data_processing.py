import torch
import random

def random_time_mask(batch_mels, max_masks=2, mask_ratio=0.2):
    """
    Randomly masks a portion of the time dimension for each mel spectrogram 
    in the batch to improve model robustness to time-domain occlusion.
    
    Args:
        batch_mels (Tensor): shape (batch_size, segment_len, feature_dim)
        max_masks (int): how many separate masks to apply per sample
        mask_ratio (float): what fraction of the time dimension to mask
    Returns:
        Tensor of shape (batch_size, segment_len, feature_dim) with time-masked regions
    """
    batch_mels_processed = []
    for mel in batch_mels:
        length = mel.size(0)
        mask_length = int(length * mask_ratio)
        masked_mel = mel.clone()
        
        for _ in range(max_masks):
            start = random.randint(0, max(0, length - mask_length))
            end = start + mask_length
            masked_mel[start:end, :] = 0.0
        
        batch_mels_processed.append(masked_mel)
    return torch.stack(batch_mels_processed, dim=0)
