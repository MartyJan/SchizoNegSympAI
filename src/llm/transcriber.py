"""
Whisper Audio Transcription Tool

Transcribe audio files using OpenAI's Whisper model.

Usage:
    # Process a single audio file with default output directory
    python transcriber.py /path/to/audio.mp3

    # Process a single audio file with custom output directory
    python transcriber.py /path/to/audio.mp3 --output-dir /path/to/output

    # Process all audio files in a directory
    python transcriber.py /path/to/audio/directory

    # Process all audio files in a directory with custom output location
    python transcriber.py /path/to/audio/directory --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path
from typing import List

import whisper

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import read_config
from core.file_utils import is_audio_file
from core.log import get_logger

config = read_config()
logger = get_logger()


def transcribe_audio(
    input_file: Path, output_file: Path, model: whisper.Whisper
) -> None:
    """
    Transcribe a single audio file and write the result to a text file.

    Args:
        input_file (Path): Path to the input audio file.
        output_file (Path): Path to save the transcription result.
        model (whisper.Whisper): Whisper model instance for transcription.
    """
    logger.info(f"Transcribing {input_file}...")
    # Add fp16=False for better punctuation accuracy
    result = model.transcribe(str(input_file), fp16=False)

    subtitles: List[str] = []
    for segment in result["segments"]:
        text = segment["text"]
        subtitles.append(text)

    # Combine all subtitles into a single string
    content = "\n".join(subtitles)

    # Write the transcript to the output file
    output_file.write_text(content, encoding="utf-8")

    logger.info(f"Transcription saved to {output_file}")


def transcribe_single(audio_path: Path, output_dir: Path) -> None:
    """
    Transcribe a single file and write output to the given directory.

    Args:
        audio_path (Path): Path to the audio file.
        output_dir (Path): Directory to save the transcript.
    """
    if not is_audio_file(audio_path):
        raise ValueError(f"'{audio_path}' is not a valid audio file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{audio_path.stem}.txt"

    model = whisper.load_model(config["llm"]["transcriber"]["whisper"])
    transcribe_audio(audio_path, output_file, model)


def transcribe_folder(audio_dir: Path, output_dir: Path) -> None:
    """
    Transcribe all audio files in a directory.

    Args:
        audio_dir (Path): Path to the directory of audio files.
        output_dir (Path): Directory to save all transcripts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(config["llm"]["transcriber"]["whisper"])

    for audio_file in sorted(audio_dir.glob("*")):
        if is_audio_file(audio_file):
            output_file = output_dir / f"{audio_file.stem}.txt"
            transcribe_audio(audio_file, output_file, model)


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
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    output_dir = Path(args.output_dir)

    if input_path.is_file():
        transcribe_single(input_path, output_dir)
    elif input_path.is_dir():
        transcribe_folder(input_path, output_dir)
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


if __name__ == "__main__":
    main()
