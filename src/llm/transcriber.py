#!/usr/bin/env python3

"""
Whisper Audio Transcription Tool

Transcribe audio files using OpenAI's Whisper model.

Usage:
    python transcriber.py path/to/audio.mp3
    python transcriber.py path/to/audio_folder --output_dir path/to/output_folder
"""

import argparse
import sys
from pathlib import Path
from typing import List

import whisper

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import read_config
from core.file_utils import is_audio_file

config = read_config()


def transcribe_audio(input_file: Path, output_file: Path) -> None:
    """
    Transcribe a single audio file and write the result to a text file.

    Args:
        input_file (Path): Path to the input audio file.
        output_file (Path): Path to save the transcription result.
    """
    model = whisper.load_model(config["llm"]["transcriber"]["whisper"])

    print(f"Transcribing {input_file}...")
    # Add fp16=False for better punctuation accuracy
    result = model.transcribe(
        str(input_file), fp16=False
    )

    subtitles: List[str] = []
    for segment in result["segments"]:
        text = segment["text"]
        subtitles.append(text)

    # Combine all subtitles into a single string
    content = "\n".join(subtitles)

    # Write the transcript to the output file
    output_file.write_text(content, encoding="utf-8")

    print(f"Transcription saved to {output_file}")


def transcribe_single(audio_path: Path, output_dir: Path) -> None:
    """
    Transcribe a single file and write output to the given directory.

    Args:
        audio_path (Path): Path to the audio file.
        output_dir (Path): Directory to save the transcript.
    """
    if not is_audio_file(audio_path):
        print(f"Error: '{audio_path}' is not a valid audio file.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{audio_path.stem}.txt"
    transcribe_audio(audio_path, output_file)


def transcribe_folder(audio_dir: Path, output_dir: Path) -> None:
    """
    Transcribe all audio files in a directory.

    Args:
        audio_dir (Path): Path to the directory of audio files.
        output_dir (Path): Directory to save all transcripts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(audio_dir.glob("*")):
        if is_audio_file(audio_file):
            output_file = output_dir / f"{audio_file.stem}.txt"
            transcribe_audio(audio_file, output_file)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI's Whisper model."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to an audio file or folder containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config["llm"]["path"]["transcript_dir"],
        help="Output directory for transcript files.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to handle transcription of audio files or directories.
    """
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    if input_path.is_file():
        transcribe_single(input_path, output_dir)
    elif input_path.is_dir():
        transcribe_folder(input_path, output_dir)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
