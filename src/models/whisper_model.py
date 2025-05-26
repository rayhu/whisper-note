"""
WhisperTranscriber class for handling speech recognition using faster-whisper.
"""

import os
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        word_timestamps: bool = True,
    ):
        """
        Initialize the Whisper model.

        Args:
            model_size (str): Size of the model to use (tiny, base, small, medium, large)
            device (str): Device to run the model on (cpu, cuda, mps)
            compute_type (str): Type of computation to use (int8, float16, float32)
            language (str, optional): Language code to use for transcription
            beam_size (int): Number of beams for beam search
            best_of (int): Number of candidates to consider
            temperature (float): Temperature for sampling
            condition_on_previous_text (bool): Whether to condition on previous text
            no_speech_threshold (float): Threshold for no speech detection
            compression_ratio_threshold (float): Threshold for compression ratio
            log_prob_threshold (float): Threshold for log probability
            word_timestamps (bool): Whether to return word timestamps
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.word_timestamps = word_timestamps

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper model.

        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code to use for transcription
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dict[str, Any]: Transcription results including text and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(
            audio_path,
            language=language or self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            word_timestamps=self.word_timestamps,
            **kwargs
        )

        return {
            "text": " ".join([segment.text for segment in segments]),
            "segments": [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ] if self.word_timestamps else None
                }
                for segment in segments
            ],
            "language": info.language,
            "language_probability": info.language_probability
        } 