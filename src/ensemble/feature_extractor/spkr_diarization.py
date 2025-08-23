"""
Speaker Diarization Tool using NVIDIA NeMo.

This script performs speaker diarization on one or more audio files.
It separates speakers and attempts to identify which speaker is the patient
(based on who starts speaking second).

Usage:
    # Process a single audio file
    python spkr_diarization.py /path/to/audio.wav

    # Process all audio files in a directory
    python spkr_diarization.py /path/to/audio/directory

    # Specify custom output directory
    python spkr_diarization.py /path/to/audio.wav --output-dir /path/to/output
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import wget
from matplotlib import pyplot as plt
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import (
    labels_to_pyannote_object,
    rttm_to_labels,
)
from omegaconf import OmegaConf
from pyannote.core.notebook import Notebook
from pydub import AudioSegment

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.config import read_config
from core.file_utils import is_audio_file
from core.log import get_logger

config = read_config()
logger = get_logger()


def audio_segment_to_numpy(sound: AudioSegment) -> np.ndarray:
    """
    Convert an AudioSegment to a NumPy array of floating-point values.

    Args:
        sound (AudioSegment): Audio segment to convert.

    Returns:
        np.ndarray: Float32 numpy array of audio data normalized between -1.0 and 1.0.
    """
    samples = [s.get_array_of_samples() for s in sound.split_to_mono()]
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr


def download_model_config(nemo_dir: Path) -> Path:
    """
    Download the diarization model config if not already available.

    Args:
        nemo_dir (Path): Directory to save the config file.

    Returns:
        Path: Path to the downloaded config file.
    """
    nemo_dir.mkdir(parents=True, exist_ok=True)
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/r1.19.0/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    return Path(wget.download(config_url, str(nemo_dir)))


def setup_model_config(
    model_config_path: Path, output_dir: Path, input_audio_path: Path
) -> OmegaConf:
    """
    Prepare and customize the model configuration for diarization.

    Args:
        model_config_path (Path): Path to the model config file.
        output_dir (Path): Output directory for diarization results.
        input_audio_path (Path): Path to the input audio file.

    Returns:
        OmegaConf: Loaded and customized model configuration object.
    """
    cfg = OmegaConf.load(model_config_path)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.diarizer.out_dir = str(output_dir)
    cfg.diarizer.manifest_filepath = str(output_dir / "input_manifest.json")
    cfg.diarizer.clustering.parameters.oracle_num_speakers = True
    cfg.diarizer.vad.parameters.onset = 0.05
    cfg.diarizer.vad.parameters.offset = 0.05
    cfg.diarizer.vad.parameters.pad_onset = 0
    cfg.diarizer.msdd_model.parameters.diar_window_length = 1000

    manifest = {
        "audio_filepath": str(input_audio_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    with open(cfg.diarizer.manifest_filepath, "w") as f:
        json.dump(manifest, f)
        f.write("\n")

    return cfg


def visualize_diarization(rttm_path: Path, output_dir: Path) -> None:
    """
    Create a visualization of diarization results and save it as an image.

    Args:
        rttm_path (Path): Path to the RTTM file with diarization results.
        output_dir (Path): Directory to save the visualization image.
    """
    pred_labels = rttm_to_labels(rttm_path)
    hypothesis = labels_to_pyannote_object(pred_labels)

    fig, ax = plt.subplots(figsize=(30, 3))
    Notebook().plot_annotation(hypothesis, ax=ax, time=True, legend=True)
    fig.savefig(output_dir / "diarization.jpg", dpi=300, bbox_inches="tight")


def process_audio_speakers(
    audio: AudioSegment, rttm_path: Path, audio_dir: Path, filename: str
) -> Tuple[str, str]:
    """
    Separate and export speaker audio tracks, then determine the patient speaker.

    Args:
        audio (AudioSegment): Full original audio segment.
        rttm_path (Path): Path to the RTTM file containing speaker timestamps.
        audio_dir (Path): Output folder for separated speaker audio files.
        filename (str): Base name of audio file without extension.

    Returns:
        Tuple[str, str]: Tuple containing (filename, identified_speaker_id),
                         where identified_speaker_id is 'spkr0', 'spkr1', or 'cannot determined'.
    """
    spkr0, spkr1 = [
        AudioSegment.silent(duration=len(audio), frame_rate=audio.frame_rate)
        for _ in range(2)
    ]

    with rttm_path.open("r") as rttm:
        for line in rttm:
            parts = line.strip().split()
            start_ms = int(float(parts[3]) * 1000)
            end_ms = int(float(parts[4]) * 1000) + start_ms
            label = parts[7]
            segment = audio[start_ms:end_ms]
            if label == "speaker_0":
                spkr0 = spkr0.overlay(segment, position=start_ms)
            elif label == "speaker_1":
                spkr1 = spkr1.overlay(segment, position=start_ms)

    spkr0_name = "spkr0"
    spkr1_name = "spkr1"
    spkr0_path = audio_dir / f"{filename}_{spkr0_name}.wav"
    spkr1_path = audio_dir / f"{filename}_{spkr1_name}.wav"
    spkr0.export(spkr0_path, format="wav")
    spkr1.export(spkr1_path, format="wav")

    # Assume the first speaker is the clinician
    duration_sec = 3
    spkr0_arr = audio_segment_to_numpy(spkr0)[: duration_sec * audio.frame_rate]
    spkr1_arr = audio_segment_to_numpy(spkr1)[: duration_sec * audio.frame_rate]
    silence0 = np.sum(spkr0_arr == 0)
    silence1 = np.sum(spkr1_arr == 0)

    if silence0 > silence1:
        dia_num = spkr0_name
    elif silence1 > silence0:
        dia_num = spkr1_name
    else:
        dia_num = "cannot determined"

    return filename, dia_num


def perform_speaker_diarization(
    input_audio_path: Path, output_base: Path
) -> Tuple[str, str]:
    """
    Perform complete speaker diarization on a single audio file.

    Args:
        input_audio_path (Path): Path to input audio file.
        output_base (Path): Base output directory for results.

    Returns:
        Tuple[str, str]: Tuple containing (filename, identified_speaker_id).
    """
    filename = input_audio_path.stem
    output_dir = output_base / filename
    audio_dir = output_dir / "audio"
    rttm_dir = output_dir / "pred_rttms"
    nemo_dir = Path(__file__).resolve().parent / ".cache" / "nemo"

    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = nemo_dir / "diar_infer_telephonic.yaml"
    if not model_config_path.exists():
        model_config_path = download_model_config(nemo_dir)

    cfg = setup_model_config(model_config_path, output_dir, input_audio_path)
    diarizer = NeuralDiarizer(cfg=cfg)
    diarizer.diarize()

    rttm_path = rttm_dir / f"{filename}.rttm"

    if not rttm_path.exists():
        logger.warning(f"RTTM file not found: {rttm_path}")
        return filename, "cannot determined"

    visualize_diarization(rttm_path, rttm_dir)

    audio = AudioSegment.from_file(input_audio_path)
    return process_audio_speakers(audio, rttm_path, audio_dir, filename)


def run_single(input_audio_path: Path, output_dir: Path, result_path: Path) -> None:
    """
    Run speaker diarization on a single audio file and write result to CSV.

    Args:
        input_audio_path (Path): Path to the input audio file.
        output_dir (Path): Output directory for results.
        result_path (Path): Path to save the result CSV file.
    """
    if not is_audio_file(input_audio_path):
        raise ValueError(f"'{input_audio_path}' is not a valid audio file.")

    filename, speaker = perform_speaker_diarization(input_audio_path, output_dir)

    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Speaker"])
        writer.writeheader()
        writer.writerow({"Filename": filename, "Speaker": speaker})


def run_folder(input_dir: Path, output_dir: Path, result_path: Path) -> None:
    """
    Run diarization on all audio files in a folder and write consolidated results to CSV.

    Args:
        input_dir (Path): Folder containing audio files.
        output_dir (Path): Output directory for results.
        result_path (Path): Path to save the consolidated results CSV file.
    """

    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Speaker"])
        writer.writeheader()

        for file in sorted(input_dir.iterdir()):
            if not is_audio_file(file):
                logger.info(f"Skipping non-audio file: {file}")
                continue

            logger.info(f"Processing: {file}")
            filename, speaker = perform_speaker_diarization(file, output_dir)
            writer.writerow({"Filename": filename, "Speaker": speaker})
            f.flush()  # Ensure each line is written immediately


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the diarization script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Speaker Diarization using NVIDIA NeMo"
    )
    parser.add_argument("input", type=str, help="Path to audio file or folder")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config["ensemble"]["path"]["audio_diarized_dir"],
        help="Directory to store outputs",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the speaker diarization process.
    """
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "diarization_results.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        run_single(input_path, output_dir, result_path)
    elif input_path.is_dir():
        run_folder(input_path, output_dir, result_path)
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


if __name__ == "__main__":
    main()
