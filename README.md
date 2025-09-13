# ML Experiments

A collection of machine learning experiments and demonstrations, with a focus on deep learning phenomena and training dynamics.

## Current Experiments

### 1. Double Descent Phenomenon (CIFAR-10)
Demonstrates the double descent phenomenon in deep learning using CIFAR-10 dataset, exploring:
- Model-wise double descent: varying model width/parameter count
- Epoch-wise double descent: training dynamics over extended periods
- Effects of label noise and dataset size on the phenomenon

## Prerequisites

- Python 3.11 or higher
- Conda (recommended for environment management)
- CUDA-capable GPU (recommended for faster training)

## Installation

1. Create and activate a conda environment:
```bash
conda env create -f environment.yml
# or
conda create -n whisper-note python=3.11
conda activate whisper-note
```

2. Install JupyterLab and kernel support:
```bash
conda install -c conda-forge jupyterlab ipykernel -y
```

3. Install PyTorch and required packages:
```bash
# For CUDA support (adjust cuda version as needed):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Other dependencies
pip install pandas matplotlib tqdm

# or

pip install -r requirements.txt

```

4. Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name whisper-note --display-name "whisper-note"
```

## Experiments

### Double Descent (double_descent.ipynb)

This notebook demonstrates the double descent phenomenon in deep learning using CIFAR-10 dataset. Features:
- Configurable model architecture (SimpleCNN with adjustable width)
- Support for label noise and reduced training set
- Experiment tracking and visualization
- Mixed precision training support
- Detailed bilingual (English/Chinese) documentation

To run the experiments:
1. Launch JupyterLab:
```bash
jupyter lab
```
2. Open `double_descent.ipynb`
3. Select the "Whisper-Note" kernel
4. Run all cells to reproduce the double descent phenomenon

## Project Structure

```
whisper-note/
├── double_descent.ipynb    # Double descent experiment notebook
├── muPT.ipynb             # Other experiments
├── requirements.txt       # Python package requirements
└── README.md             # This file
```




A Python-based audio transcription service using OpenAI's Whisper model.

### Whisper Audio Transcription

- Real-time audio recording and transcription
- Support for multiple output formats (JSON, TXT, SRT)
- Configurable model parameters
- Simple command-line interface

Command Line Interface

1. Record and transcribe audio:
```bash
python src/main.py --mode record
```

2. Transcribe an existing audio file:
```bash
python src/main.py --mode transcribe --input path/to/audio.wav
```

3. Specify output format:
```bash
python src/main.py --mode transcribe --input path/to/audio.wav --output-format srt
```

### Output Formats

- JSON: Full transcription with metadata and timestamps
- TXT: Plain text transcription
- SRT: Subtitle format with timestamps

## Configuration

Configuration settings can be modified in `src/config/settings.py`:

- Model settings (size, device, compute type)
- Audio settings (sample rate, channels)
- Output settings (directory, format)

## Project Structure

```
whisper-note/
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   └── whisper_model.py
│   ├── services/
│   │   └── audio_service.py
│   └── main.py
├── tests/
├── data/
│   └── output/
├── requirements.txt
└── README.md
```

## License

Copyright 2025 Ray Hu.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0