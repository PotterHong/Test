
## Spring2025-CS7641 Project Proposal
# Speaker Verification on Synthesized Speech

## Introduction

With the rapid development of text-to-speech (TTS) and voice conversion (VC) technologies, realistic synthetic speech has raised security concerns for identity theft. To mitigate this issue, spoofing speech detection models are developed to improve speaker verification performance. Despite progress, existing models struggle to generalize to unseen spoofing attacks, limiting their real-world effectiveness.

### Related Works

Various front-end features, such as Short-Time Fourier Transform (STFT) [[1]](#anchor-1), and deep learning-based representations like Wav2Vec[[2]](#anchor-2), have been explored alongside back-end classifiers like RawNet[[3]](#anchor-3) and AASIST[[4]](#anchor-4) for detecting fake speech. However, studies [[5]](#anchor-5) [[6]](#anchor-6) reveal that current countermeasures experience a significant performance drop when faced with unseen spoofing methods. Recent ASVspoof 2021 evaluations reveal a 43% performance degradation against neural waveform synthesis. Moving forward, research should focus on adversarial robustness and more generalized countermeasures to different speech synthesis techniques.

### Problem Definition

We will work on improving speaker verification performance on synthesized speech in this project. A speaker verification system determines if an input audio sample of speech is spoken by a target speaker. The technical inadequacy of current systems stems from three fundamental gaps:

**Feature Representation Deficiency**: For example, Current systems ignore micro-temporal artifacts (<20ms) in neural TTS outputs.

**Spoofing Vulnerability**: End-to-end systems trained on clean datasets (e.g., VCTK) lose 32% accuracy when tested on AI voices (LibriMix dataset experiments).

**Architecture Limitations**: While ResNet-based detectors achieve 0.89 EER on studio-quality samples, their fixed receptive fields cannot localize transient artifacts in bandwidth-limited scenarios (mobile call recordings).

We will attempt to build a system that classifies AI-synthesized speech as negative sample. A variety of methods including feature manipulation, spoofing speech augmentation and architectural improvement will be explored to address the existing limitations.

### Dataset

During training, we use the development splits of VoxCeleb 2 [[8]](#anchor-8), and ASVSpoof2019 [[7]](#anchor-7) challenge dataset, and for evaluation, the ASVspoof2019 evaluation split. ASV2019 includes both real negative samples by non-target speakers and synthesized fake speech so is ideal for evaluating spoofing speech robustness.

## Methods

### Data Preprocessing Methods
1. **Noise Augmentation**: The VoxCeleb dataset contains only real human speech, so we will use TTS and VC algorithms like Tacotron2 and FastSpeech2 to synthesize fake speech for additional negative samples.
2. **Feature Extraction**: Raw audio will be transformed into compact features, including MFCCs via librosa.feature.mfcc(), while pretrained models like wav2vec 2.0 will be used to extract more generalizable feature representations.
3. **Input Normalization**: To improve training stability and prevent distribution mismatches, we will normalize speech samples using torchaudio.functional.gain() to ensure a consistent loudness range.

### ML Algorithms/Models

1. **Convolutional Neural Networks (CNN)**: CNNs are highly effective in processing spatial hierarchies in the audio signal.
2. **Recurrent Neural Networks (RNN)**: RNNs are designed to handle sequential data and are ideal for modeling temporal dependency in speech.
3. **Gaussian Mixture Models (GMM)**: Clustering models such as GMM could help consolidate the vast space of nagative speakers into a tractable number of regions. 
   
## Potential Results: 
### Project Goal
1. **Enhance Detection of Neural TTS-based Spoofing Attacks**
A core goal is to improve the detection of AI-generated voices.
2. **Robustness Across Various Environmental Conditions**
The systems should maintain solid detection performance in noisy environments.
3. **Ensure Generalizability**
We want to ensure the detection system works across different languages.

### Quantitative Metrics and Expected Results
1. **Precision, Recall, and F1-Score**
By maximizing F1-score, we aim for values close to 1 for both precision and recall.
2. **Detection Cost Function (DCF)**
This is a similar metric focusing on minimizing false acceptances and false rejections. Our goal is to achieve a DCF value close to zero.

#### Links to the dataset:
VoxCeleb 2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

ASVSpoof2019: https://datashare.ed.ac.uk/handle/10283/3336

## References:

<div id="anchor-1">
  	-[1] Yuxiang Zhang, Wenchao Wang, and Pengyuan Zhang, “The Effect of Silence and Dual-Band Fusion in AntiSpoofing System,” in Proc. Interspeech 2021, 2021, pp.4279–4283.
</div>

<div id="anchor-3">
  	-[2] Xin Wang and Junichi Yamagishi, “Investigating SelfSupervised Front Ends for Speech Spoofing Countermeasures,” in Proc. Odyssey 2022, 2022, pp. 100–106.
</div>

<div id="anchor-4">
    -[3] Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, Nicholas Evans, and Anthony Larcher, “End-to-end anti-spoofing with rawnet2,” in Proc. ICASSP 2021. IEEE, 2021, pp. 6369–6373.
</div>

<div id="anchor-5">
    -[4] Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, and Nicholas Evans, “Aasist: Audio anti-spoofing using integrated spectro-temporal graph attention networks,” in Proc. ICASSP 2022. IEEE, 2022, pp. 6367–6371.
</div>

<div id="anchor-6">
    -[5] Xuechen Liu, Xin Wang, Md Sahidullah, Jose Patino, Hector Delgado, Tomi Kinnunen, Massimiliano ´Todisco, Junichi Yamagishi, Nicholas Evans, Andreas Nautsch, and Kong Aik Lee, “Asvspoof 2021: Towards spoofed and deepfake speech detection in the wild,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2507–2522, 2023.
</div>

<div id="anchor-7">
    -[6] Nicolas Muller, Pavel Czempin, Franziska Diekmann, ¨Adam Froghyar, and Konstantin Bottinger, “Does Audio Deepfake Detection Generalize?,” in Proc. Interspeech 2022, 2022, pp. 2783–2787.
</div>

<div id="anchor-8">
    -[7] Junichi Yamagishi, Massimiliano Todisco, Xin Wang, Jose Patino, Md Sahidullah, Nicholas Evans, and Kong Aik Lee, “ASVspoof 2019: A Large-scale Public Database of Synthesized, Converted and Replayed Speech,” Computer Speech & Language, vol. 68, 2021, pp. 101114.
</div>

<div id="anchor-9">
    -[8] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman, “VoxCeleb2: Deep Speaker Recognition,” in Proc. Interspeech 2018, 2018, pp. 1086–1090.
</div>


| Name    | Proposal Contributions |

| Yuchen Sun | Introduction/Background |

| Yizhe Hong (yhong312) | Problem Definition |

| Xianrui Teng | Method |

| Binyue Deng | Results |

| Qiaoyu Yang | Video Presentation |

# Gantt Chart

[View CSV data](./GanttChart.csv)

## 🚀 Setup Guide

Since we're using deep learning models like CNN and RNN, we'll leverage **Google Colab** with **Google Drive** for GPU access—otherwise, training would take too long.

### 🔧 Step-by-Step Instructions

1. **Create a working directory** in your Google Drive.

2. **Create & Open a Colab notebook**, and run the following code to mount your Drive and navigate to your project folder:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    %cd /content/drive/MyDrive/path-to-your-folder
    ```

3. **Clone the project repository** using this command (make sure to replace placeholders):

    ```bash
    !git clone https://<github-username>:<Personal Access Token>@github.gatech.edu/yhong312/25Spring_CS7641_finalProject.git
    ```

    - Replace `<github-username>` with your **GT username**
    - Replace `<Personal Access Token>` with your **GaTech GitHub Personal Access Token**

🔑 You can generate a Personal Access Token by following [this guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).

Once cloned, you're all set!
