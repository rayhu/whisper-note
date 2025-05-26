"""
Configuration settings for the Whisper transcription service.
"""

import os
from typing import Dict, Any

# Model settings
MODEL_SETTINGS: Dict[str, Any] = {
    "model_size": "base",  # tiny, base, small, medium, large
    "device": "cpu",  # cpu, cuda, mps
    "compute_type": "int8",  # int8, float16, float32
    "language": None,  # None for auto-detection
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,
    "condition_on_previous_text": True,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "word_timestamps": True,
}

# Audio settings
AUDIO_SETTINGS: Dict[str, Any] = {
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav",
}

# Output settings
OUTPUT_SETTINGS: Dict[str, Any] = {
    "output_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "output"),
    "save_format": "json",  # json, txt, srt
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_SETTINGS["output_dir"], exist_ok=True) 