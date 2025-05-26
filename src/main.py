"""
Command-line interface for the audio transcription service.
"""

import os
import sys
import argparse
from typing import Optional

from services.audio_service import AudioService

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Audio Transcription Service")
    parser.add_argument(
        "--mode",
        choices=["record", "transcribe"],
        required=True,
        help="Operation mode: record audio or transcribe existing file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input audio file (required for transcribe mode)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "txt", "srt"],
        default="json",
        help="Output format for transcription (default: json)"
    )
    return parser.parse_args()

def main():
    """
    Main entry point for the application.
    """
    args = parse_args()
    service = AudioService()

    if args.mode == "record":
        print("Recording started. Press Ctrl+C to stop...")
        try:
            service.start_recording()
            while True:
                pass
        except KeyboardInterrupt:
            print("\nRecording stopped.")
            audio_path = service.stop_recording()
            if audio_path:
                print(f"Audio saved to: {audio_path}")
                print("Transcribing...")
                transcription = service.transcribe_audio(audio_path)
                output_path = service.save_transcription(transcription, audio_path)
                print(f"Transcription saved to: {output_path}")

    elif args.mode == "transcribe":
        if not args.input:
            print("Error: --input argument is required for transcribe mode")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        print(f"Transcribing {args.input}...")
        transcription = service.transcribe_audio(args.input)
        output_path = service.save_transcription(transcription, args.input)
        print(f"Transcription saved to: {output_path}")

if __name__ == "__main__":
    main() 