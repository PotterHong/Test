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

### Data Preprocessing Methods
1. **Feature Extraction (MFCC & Spectral Features)**: Extracting MFCCs or spectrograms captures key speech characteristics. This transforms raw audio into machine-readable features.
2. **Noise Reduction**: Audio data collected from real-world environments often contains background noise, interference, or artefacts.
3. **Voice Activity Detection (VAD)**: VAD isolates speech segments, removing silence and irrelevant noise, ensuring only useful audio is processed.
4. **Data Augmentation**: such as pitch shifting or adding noise, expands the dataset, improving model robustness in varying conditions.
5. **Normalization**: this standardizes feature scales, aiding in stable and efficient training.

All the above data preprocessing methods are supported by `librosa`.

### ML Algorithms/Models

1. **Convolutional Neural Networks (CNN)**: CNNs are highly effective in processing grid-like data, such as spectrograms. They can capture spatial hierarchies in the audio signal. Use `Conv2d`, `MaxPool2d`, `ReLU`, and `Sequential` from `torch.nn`.
2. **Recurrent Neural Networks (RNN) / Long Short-Term Memory (LSTM)**: RNNs and LSTMs are designed to handle sequential data and are ideal for modeling time-series data like audio. They can capture temporal dependencies in speech signals. Use `LSTM` and `RNN` from `torch.nn`.
3. **Support Vector Machines (SVM)**: SVMs are a classical model for binary classification, and they should work well with high-dimensional feature spaces such as those from MFCCs. Use `SVC` from `sklearn.svm`.
4. **Gradient Boosting Machines (GBM) / XGBoost**: GBM and XGBoost are powerful tree-based models that can be used for classification tasks. Use `GradientBoostingClassifier` from `sklearn.ensemble`.
5. **Random Forest**: Random Forest is an ensemble method based on decision trees. It works well for classification tasks and is robust against overfitting, makeing it worth to try. Use `RandomForestClassifier` from `sklearn.ensemble`.

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


