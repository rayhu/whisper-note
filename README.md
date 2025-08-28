# Whisper Note

A Python-based audio transcription service using OpenAI's Whisper model.

## Features

- Real-time audio recording and transcription
- Support for multiple output formats (JSON, TXT, SRT)
- Configurable model parameters
- Simple command-line interface

## Prerequisites

- Python 3.11 or higher
- Conda (recommended for environment management)

## Installation

1. Create and activate a conda environment:
```bash
conda create -n whisper-note python=3.11
conda activate whisper-note
```

2. Install JupyterLab and kernel support:
```bash
conda install -c conda-forge jupyterlab ipykernel -y
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name whisper-note --display-name "whisper-note"
```

Export the environment for Conda
```
conda env export --name whisper-note --from-history > environment.yml
```

## Usage

### Command Line Interface

1. Record and transcribe audio:
```bash
python src/main.py --mode record
How t```

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