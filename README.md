# Moss Audio Tokenizer
<div align="center">
<p>
<img src="images/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
<p>

<img src="https://img.shields.io/badge/MossAudio-Tokenizer-ee4c2c?style=flat&logo=soundcharts&logoColor=white" alt="MossAudioTokenizer"/>
<img src="https://img.shields.io/badge/Semantic--aware-Logic-3776ab?style=flat&logo=probot&logoColor=white" alt="Semantic-aware"/>
<img src="https://img.shields.io/badge/Transformer-Architecture-555555?style=flat&logo=micro-dot-blog&logoColor=white" alt="Transformer"/>
<img src="https://img.shields.io/badge/Modal-Speech-e1b12c?style=flat&logo=google-assistant&logoColor=white" alt="Speech"/>
<img src="https://img.shields.io/badge/Modal-Audio-4b8bbe?style=flat&logo=audiomack&logoColor=white" alt="Audio"/>
<img src="https://img.shields.io/badge/Modal-Music-f39c12?style=flat&logo=apple-music&logoColor=white" alt="Music"/>


<!-- TODO: replace the paper link to the arXiv link -->
[![arXiv](https://img.shields.io/badge/arXiv-MOSSAudioTokenizer.svg)](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.git)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
</div>

## Introduction
**MOSS Audio Tokenizer** is a unified discrete audio tokenizer based on the **Cat** (**C**ausal **A**udio **T**okenizer with **T**ransformer) architecture. Scaling to 1.6 billion parameters, it functions as a unified discrete interface, delivering both lossless-quality reconstruction and high-level semantic alignment.

**Key Features:**

*   **Extreme Compression & Variable Bitrate**: It compresses 24kHz raw audio into a remarkably low frame rate of 12.5Hz. Utilizing a 32-layer Residual Vector Quantizer (RVQ), it supports high-fidelity reconstruction across a wide range of bitrates, from 0.125kbps to 4kbps.
*   **Pure Transformer Architecture**: The model features a "CNN-free" homogeneous architecture built entirely from Causal Transformer blocks. With 1.6B combined parameters (Encoder + Decoder), it ensures exceptional scalability and supports low-latency streaming inference.
*   **Large-Scale General Audio Training**: Trained on 3 million hours of diverse audio data, the model excels at encoding and reconstructing all audio domains, including speech, sound effects, and music.
*   **Unified Semantic-Acoustic Representation**: While achieving state-of-the-art reconstruction quality, Cat produces discrete tokens that are "semantic-rich," making them ideal for downstream tasks like speech understanding (ASR) and generation (TTS).
*   **Fully Trained From Scratch**: Cat does not rely on any pretrained encoders (such as HuBERT or Whisper) or distillation from teacher models. All representations are learned autonomously from raw data.
*   **End-to-End Joint Optimization**: All components—including the encoder, quantizer, decoder, discriminator, and a decoder-only LLM for semantic alignment—are optimized jointly in a single unified training pipeline.

**Summary:**
By combining a simple, scalable architecture with massive-scale data, the Cat architecture overcomes the bottlenecks of traditional audio tokenizers. It provides a robust, high-fidelity, and semantically grounded interface for the next generation of native audio foundation models.



This repository is the official implementation of Moss Audio Tokenizer.

<br>
<p align="center">
    <img src="images/arch.png" width="95%"> <br>
    Architecture of Moss Audio Tokenizer
</p>
<br>

## Qick Link
* [Release](#release)
* [Installation](#installation)
* [Model List](#model-list)
* [Usage](#usage)
* [Quick Start](#quick-start)
* [Evaluation Metrics](#evaluation-metrics)
* [Repository layout](#repository-layout)
* [Citation](#citation)
* [License](#license)

## Release
<!-- TODO: replace the paper link to the arXiv link -->
- [2026/2/9] 🔥 We released code and checkpoints of Moss Audio Tokenizer. Checkout the [paper](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.git) and [model_weights](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer).
- [ ] 🚧 Evaluation scripts

## Installation

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.git
cd MOSS-Audio-Tokenizer
pip install -r requirements.txt
```

## Model List
### 🎵 Moss Audio Tokenizer
| Model | 🤗 Hugging Face |
|:-----:|:---------------:|
| **🚀 Moss Audio Tokenizer** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) |

### 🎵 Audio Generation Models Based On Moss Audio Tokenizer
| Model | 🤗 Hugging Face |
|:-----:|:---------------:|
| **🚀 Moss-TTS** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) |
| **🚀 Moss-TTSD** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) |
| **🚀 MOSS-TTS-Local-Transformer** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) |
| **🚀 MOSS-TTS-Realtime** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) |
| **🚀 MOSS-Voice-Generator** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-Voice-Generator) |
| **🚀 MOSS-SoundEffect** | [![HF](https://img.shields.io/badge/🤗%20HuggingFace-MOSSAudioTokenizer-yellow)](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) |


## Usage

### Reconstruction

```python
import torch
from transformers import AutoModel
import torchaudio

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

wav, sr = torchaudio.load('demo/demo_gt.wav')
if sr != model.sampling_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sampling_rate)
wav = wav.unsqueeze(0)
enc = model.encode(wav, return_dict=True)
print(f"enc.audio_codes.shape: {enc.audio_codes.shape}")
dec = model.decode(enc.audio_codes, return_dict=True)
print(f"dec.audio.shape: {dec.audio.shape}")
wav = dec.audio.squeeze(0)
torchaudio.save("demo/demo_rec.wav", wav, sample_rate=model.sampling_rate)

# Decode using only the first 8 layers of the RVQ
dec_rvq8 = model.decode(enc.audio_codes[:8], return_dict=True)
wav_rvq8 = dec_rvq8.audio.squeeze(0)
torchaudio.save("demo/demo_rec_rvq8.wav", wav_rvq8, sample_rate=model.sampling_rate)
```

### Streaming

`MossAudioTokenizerModel.encode` and `MossAudioTokenizerModel.decode` support simple streaming via a `chunk_duration`
argument.

- `chunk_duration` is expressed in seconds.
- It must be <= `MossAudioTokenizerConfig.causal_transformer_context_duration`.
- `chunk_duration * MossAudioTokenizerConfig.sampling_rate` must be divisible by `MossAudioTokenizerConfig.downsample_rate`.
- Streaming chunking only supports `batch_size=1`.

```python
import torch
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
audio = torch.randn(1, 1, 3200)  # dummy waveform

# 0.08s @ 24kHz = 1920 samples, divisible by downsample_rate=1920
enc = model.encode(audio, return_dict=True, chunk_duration=0.08)
dec = model.decode(enc.audio_codes, return_dict=True, chunk_duration=0.08)
```

## Quick Start

### Installation

#### Conda Linux

```bash
# Clone the repository
git clone https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.git
cd MOSS-Audio-Tokenizer

# Install dependencies
conda create -n moss-audio-tokenizer python=3.10 -y
conda activate moss-audio-tokenizer
pip install -r requirements.txt
```

### Quick Test

#### Loading Model
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("OpenMOSS-Team/MOSS-Audio-Tokenizer", trust_remote_code=True).eval()
```

#### Testing Model
```bash
cd MOSS-Audio-Tokenizer
conda activate moss-audio-tokenizer
python demo/test_reconstruction.py
```

## Repository layout

- `configuration_moss_audio_tokenizer.py`
- `modeling_moss_audio_tokenizer.py`
- `__init__.py`
- `config.json`
- model weights

## Evaluation Metrics

The table below compares the reconstruction quality of open-source audio tokenizers with Moss Audio Tokenizer on speech and audio/music data.

- Speech metrics are evaluated on LibriSpeech test-clean (English) and AISHELL-2 (Chinese), reported as EN/ZH.
- Audio metrics are evaluated on the AudioSet evaluation subset, while music metrics are evaluated on MUSDB, reported as audio/music.
- STFT-Dist. denotes the STFT distance.
- Higher is better for speech metrics, while lower is better for audio/music metrics (Mel-Loss, STFT-Dist.).
- Nvq denotes the number of quantizers.

<br>
<p align="center">
    <img src="images/reconstruct_comparison_table.png" width="95%"> <br>
    Reconstruction quality comparison of open-source audio tokenizers on speech and audio/music data.
</p>
<br>

### LibriSpeech Speech Metrics (MOSS Audio Tokenizer vs. Open-source Tokenizers)

The plots below compare our MOSS Audio Tokenizer model with other open-source speech tokenizers on the LibriSpeech dataset, evaluated with SIM, STOI, PESQ-NB, and PESQ-WB (higher is better).
We control the bps of the same model by adjusting the number of RVQ codebooks used during inference.

<br>
<p align="center">
    <img src="images/metrics_on_librispeech_test_clean.png" width="100%"> <br>
</p>
<br>



## Citation
If you use this code or result in your paper, please cite our work as:
```tex

```


## License
<!-- TODO: check and add license -->
Moss Audio Tokenizer is released under the Apache 2.0 license.