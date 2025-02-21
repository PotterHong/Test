# 25Spring_CS7641_finalProject
## Introduction/Background: Provide an introduction of your topic and literature review of related work. Briefly explain your dataset and its features, and provide a link to the dataset if possible.

### Spoofing Speech Detection

With the rapid development of **text-to-speech (TTS)** and **voice conversion (VC)** technologies, generating realistic synthetic speech has become cheaper, raising security concerns for **automatic speaker verification (ASV)** systems. To mitigate this issue, researchers have developed **spoofing speech detection** models, typically consisting of a two-part architecture with a front-end feature extractor and a back-end classifier. Despite progress, a key challenge remains—existing models struggle to generalize to unseen spoofing attacks, varying recording conditions, and compression artifacts, limiting their real-world effectiveness.

### Literature Review

Various front-end features, such as **Short-Time Fourier Transform (STFT)** [[1]](#anchor-1), **Constant Q Cepstral Coefficients (CQCC)**[[2]](#anchor-2), and deep learning-based representations like **Wav2Vec**[[3]](#anchor-3), have been explored alongside back-end classifiers like **RawNet**[[4]](#anchor-4) and **AASIST**[[5]](#anchor-5) for detecting fake speech. However, studies [[6]](#anchor-6) [[7]](#anchor-7) reveal that current countermeasures experience a significant performance drop when faced with unseen spoofing methods, adversarial attacks, or different transmission channels.

Additionally, recent efforts include open-set evaluation, allowing models to leverage external data and pre-trained speech foundation models for improved generalization. New evaluation metrics, such as **minimum detection cost function (minDCF)**, **log-likelihood-ratio cost function (Cllr)**, and **architecture-agnostic DCF (a-DCF)**[[8]](#anchor-8), have been introduced to better assessment both detection and calibration performance. Moving forward, research should focus on adversarial robustness, adaptive learning techniques, and more generalized countermeasures to enhance security in **ASV** systems.

### Dataset

We use **ASVspoof 5**, the  fifth edition of the **ASVspoof** challenge series. it advances **speech spoofing detection** and **deepfake security** with two tracks: standalone spoofing detection and spoofing-robust **ASV (SASV)**. It introduces a crowdsourced dataset from the **Multilingual LibriSpeech (MLS)** corpus, featuring more speakers, diverse acoustics, and stronger spoofing attacks, including **TTS**, **VC**, and adversarial attacks. A key innovation of **ASVspoof 5** is the open condition, allowing external data and pre-trained models. New metrics like **minDCF** and **a-DCF** also enhance assessment.

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

<div id="anchor-1">
  	-[1] Yuxiang Zhang, Wenchao Wang, and Pengyuan Zhang, “The Effect of Silence and Dual-Band Fusion in AntiSpoofing System,” in Proc. Interspeech 2021, 2021, pp.4279–4283.
</div>

<div id="anchor-2">
  	-[2] Massimiliano Todisco, Hector Delgado, and Nicholas ´ Evans, “Constant q cepstral coefficients: A spoofing countermeasure for automatic speaker verification,” Computer Speech & Language, vol. 45, pp. 516–535, 2017.
</div>

<div id="anchor-3">
  	-[3] Xin Wang and Junichi Yamagishi, “Investigating SelfSupervised Front Ends for Speech Spoofing Countermeasures,” in Proc. Odyssey 2022, 2022, pp. 100–106.
</div>

<div id="anchor-4">
    -[4] Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, Nicholas Evans, and Anthony Larcher, “End-to-end anti-spoofing with rawnet2,” in Proc. ICASSP 2021. IEEE, 2021, pp. 6369–6373.
</div>

<div id="anchor-5">
    -[5] Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, and Nicholas Evans, “Aasist: Audio anti-spoofing using integrated spectro-temporal graph attention networks,” in Proc. ICASSP 2022. IEEE, 2022, pp. 6367–6371.
</div>

<div id="anchor-6">
    -[6] Xuechen Liu, Xin Wang, Md Sahidullah, Jose Patino, Hector Delgado, Tomi Kinnunen, Massimiliano ´Todisco, Junichi Yamagishi, Nicholas Evans, Andreas Nautsch, and Kong Aik Lee, “Asvspoof 2021: Towards spoofed and deepfake speech detection in the wild,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2507–2522, 2023.
</div>

<div id="anchor-7">
    -[7] Nicolas Muller, Pavel Czempin, Franziska Diekmann, ¨Adam Froghyar, and Konstantin Bottinger, “Does Audio Deepfake Detection Generalize?,” in Proc. Interspeech 2022, 2022, pp. 2783–2787.
</div>

<div id="anchor-8">
    -[8] Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, and Nicholas Evans, “Rawboost: A raw data boosting and augmentation method applied to automatic speaker verification anti-spoofing,” in Proc. ICASSP 2022, 2022, pp. 6382–6386.
</div>

https://arxiv.org/abs/2210.02437   
https://github.com/asvspoof-challenge/2021   
https://ieeexplore.ieee.org/document/9052938   

# Dataset Link
https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.pdf   
https://github.com/JorisCos/LibriMix   
https://datashare.ed.ac.uk/handle/10283/3443   

| Name    | Proposal Contributions |

| Yuchen Sun | Introduction/Background |

| Yizhe Hong (yhong312) | Problem Definition |

| Binyue Deng | Results |

| Qiaoyu Yang | Video Presentation |

