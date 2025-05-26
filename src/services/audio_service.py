"""
AudioService class for handling audio recording and transcription.
"""

import os
import json
import wave
import pyaudio
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..models.whisper_model import WhisperTranscriber
from ..config.settings import MODEL_SETTINGS, AUDIO_SETTINGS, OUTPUT_SETTINGS

class AudioService:
    def __init__(self):
        """
        Initialize the audio service with Whisper model and PyAudio.
        """
        self.whisper = WhisperTranscriber(**MODEL_SETTINGS)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames: List[bytes] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """
        Start recording audio from the microphone.
        """
        if self.is_recording:
            return

        self.frames = []
        self.is_recording = True

        def callback(in_data, frame_count, time_info, status):
            if self.is_recording:
                self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        self.stream = self.audio.open(
            format=self.audio.get_format_from_width(2),  # 16-bit
            channels=AUDIO_SETTINGS["channels"],
            rate=AUDIO_SETTINGS["sample_rate"],
            input=True,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        self.stream.start_stream()

    def stop_recording(self) -> str:
        """
        Stop recording and save the audio to a file.

        Returns:
            str: Path to the saved audio file
        """
        if not self.is_recording:
            return ""

        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.{AUDIO_SETTINGS['format']}"
        filepath = os.path.join(OUTPUT_SETTINGS["output_dir"], filename)

        # Save audio to file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(AUDIO_SETTINGS["channels"])
            wf.setsampwidth(self.audio.get_sample_size(self.audio.get_format_from_width(2)))
            wf.setframerate(AUDIO_SETTINGS["sample_rate"])
            wf.writeframes(b''.join(self.frames))

        return filepath

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file using Whisper.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            Dict[str, Any]: Transcription results
        """
        return self.whisper.transcribe(audio_path)

    def save_transcription(self, transcription: Dict[str, Any], audio_path: str) -> str:
        """
        Save transcription results to a file.

        Args:
            transcription (Dict[str, Any]): Transcription results
            audio_path (str): Path to the original audio file

        Returns:
            str: Path to the saved transcription file
        """
        # Generate output filename based on audio filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(
            OUTPUT_SETTINGS["output_dir"],
            f"{base_name}_transcription.{OUTPUT_SETTINGS['save_format']}"
        )

        if OUTPUT_SETTINGS["save_format"] == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
        elif OUTPUT_SETTINGS["save_format"] == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription["text"])
        elif OUTPUT_SETTINGS["save_format"] == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(transcription["segments"], 1):
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    f.write(f"{i}\n{start} --> {end}\n{segment['text']}\n\n")

        return output_path

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate() 