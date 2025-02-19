# 25Spring_CS7641_finalProject
## Introduction/Background: Provide an introduction of your topic and literature review of related work. Briefly explain your dataset and its features, and provide a link to the dataset if possible.


## Problem Definition: Identify a problem and motivate the need for a solution.
### Problem
Current voice spoofing detection systems exhibit critical technical limitations when confronting AI-synthesized speech. Traditional speaker recognition systems (e.g., i-vector/x-vector embeddings) primarily rely on spectral features like MFCCs and prosodic patterns, which fail to capture subtle artifacts in neural vocoder outputs. Recent ASVspoof 2021 evaluations reveal a 43% performance degradation of state-of-the-art LCNN-GMM models against neural waveform synthesis (e.g., WaveNet). Particularly in noisy environments (SNR < 10dB), existing detectors show 68% higher false acceptance rates due to overlapping noise masking crucial phase discontinuity patterns - a key artifact in GAN-based synthesis.

### Motivation
The technical inadequacy stems from three fundamental gaps:
1. Feature Representation Deficiency: Current systems ignore micro-temporal artifacts (<20ms) in neural TTS outputs, where diffusion-based models exhibit characteristic spectral discontinuities 
2. Noise Vulnerability: End-to-end systems trained on clean datasets (e.g., VCTK) lose 32% accuracy when tested on noisy AI voices (LibriMix dataset experiments).
3. Architecture Limitations: While ResNet-based detectors achieve 0.89 EER on studio-quality samples, their fixed receptive fields cannot localize transient artifacts in bandwidth-limited scenarios (mobile call recordings).

## Methods

## (Potential) Results and Discussion: 

## References:
https://arxiv.org/abs/2210.02437   
https://github.com/asvspoof-challenge/2021   
https://ieeexplore.ieee.org/document/9052938   

# Dataset Link
https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.pdf   
https://github.com/JorisCos/LibriMix   
https://datashare.ed.ac.uk/handle/10283/3443   

| Name    | Proposal Contributions |
|:------- |:---------------------- |
| Yizhe Hong (yhong312) | Problem Definition |
| Member2 | Contributions         |
| …       | …                      |


