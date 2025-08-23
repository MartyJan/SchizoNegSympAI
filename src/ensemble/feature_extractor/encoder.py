"""
Feature Extraction Module for Multimodal Analysis

This script extracts features from audio, facial landmarks, and pose data
for further processing and analysis in machine learning pipelines.

Usage:
    python encoder.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.config import read_config
from core.log import get_logger

logger = get_logger()


class FeatureExtractor:
    """
    Extracts and processes multimodal features from audio and video data.

    This class handles feature extraction from facial landmarks, pose data,
    and audio waveforms, preparing them for machine learning models.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        audio_model: torch.nn.Module,
        device: str,
        audio_fps: int,
        video_fps: int,
    ) -> None:
        """
        Initialize the feature extractor with configuration and models.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing processing parameters
            audio_model (torch.nn.Module): Pre-trained audio model for feature extraction
            device (str): Processing device (CPU or CUDA)
            audio_fps (int): Audio sampling rate in frames per second
            video_fps (int): Video frame rate in frames per second
        """
        self.config = config
        self.audio_model = audio_model
        self.device = device
        self.video_fps = video_fps
        self.audio_fps = audio_fps

        self.n_video_frames = int(
            config["ensemble"]["encoder"]["video_window_len_in_sec"] * video_fps
        )
        self.n_audio_frames = int(
            config["ensemble"]["encoder"]["audio_window_len_in_sec"] * audio_fps
        )

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize numpy array to range [-1, 1].

        Args:
            x (np.ndarray): Input numpy array to normalize

        Returns:
            np.ndarray: Normalized numpy array with values between -1 and 1
        """
        x_min = np.min(x)
        x_max = np.max(x)
        return -1 + 2 * (x - x_min) / (x_max - x_min) if x_max > x_min else x

    def get_facial_feature(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract facial features from landmark dataframe.

        Processes facial blendshape data in windows, computing statistics
        for each valid window segment.

        Args:
            df (pd.DataFrame): DataFrame containing facial landmark data

        Returns:
            np.ndarray: Array of facial features with mean and standard deviation for each window
        """
        BSHAPE_START_IDX = df.columns.get_loc("blendShapes._neutral")
        BSHAPE_END_IDX = df.columns.get_loc("blendShapes.noseSneerRight")
        assert (BSHAPE_END_IDX - BSHAPE_START_IDX + 1) == 52, (
            "Blendshape columns are incorrect"
        )

        facial_wnd_arr: List[np.ndarray] = []
        start = 0
        while (end := start + self.n_video_frames) < len(df):
            wnd = df.iloc[start:end]
            wnd_high_conf = wnd[wnd["faceDetected"] > 0]
            min_ratio = self.config["ensemble"]["encoder"][
                "min_valid_video_frame_ratio"
            ]

            if len(wnd_high_conf) / len(wnd) >= min_ratio:
                blendshapes = wnd_high_conf.iloc[
                    :, BSHAPE_START_IDX : BSHAPE_END_IDX + 1
                ]
                mean = blendshapes.mean(axis=0).to_numpy()
                std = blendshapes.std(axis=0).to_numpy()
                facial_wnd_arr.append(np.concatenate([mean, std]))

            start += self.n_video_frames

        return np.stack(facial_wnd_arr) if facial_wnd_arr else np.array([])

    def get_pose_feature(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract pose features from landmark dataframe.

        Processes pose coordinate data in windows, computing statistics
        for each valid window segment.

        Args:
            df (pd.DataFrame): DataFrame containing pose landmark data

        Returns:
            np.ndarray: Array of pose features with mean and standard deviation for each window
        """
        POSE_X_START_IDX = df.columns.get_loc("pose.0.x")
        POSE_X_END_IDX = df.columns.get_loc("pose.24.x")
        assert (POSE_X_END_IDX - POSE_X_START_IDX) == 24 * 4, (
            "Pose columns are incorrect"
        )

        wnd_arr: List[np.ndarray] = []
        start = 0
        while (end := start + self.n_video_frames) < len(df):
            wnd = df.iloc[start:end]
            wnd_high_conf = wnd[wnd["poseDetected"] > 0]
            min_ratio = self.config["ensemble"]["encoder"][
                "min_valid_video_frame_ratio"
            ]

            if len(wnd_high_conf) / len(wnd) >= min_ratio:
                # Extract x, y, z coordinates
                x = wnd_high_conf.iloc[:, POSE_X_START_IDX : (POSE_X_END_IDX + 1) : 4]
                y = wnd_high_conf.iloc[
                    :, (POSE_X_START_IDX + 1) : (POSE_X_END_IDX + 2) : 4
                ]
                z = wnd_high_conf.iloc[
                    :, (POSE_X_START_IDX + 2) : (POSE_X_END_IDX + 3) : 4
                ]

                coords = pd.concat([x, y, z], axis=1)
                mean = coords.mean(axis=0).to_numpy()
                std = coords.std(axis=0).to_numpy()
                wnd_arr.append(np.concatenate([mean, std]))

            start += self.n_video_frames

        return np.stack(wnd_arr) if wnd_arr else np.array([])

    def get_audio_feature(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Extract audio features from waveform using the audio model.

        Processes audio data in windows, extracting features using
        the provided audio model.

        Args:
            waveform (torch.Tensor): Audio waveform torch.Tensor

        Returns:
            np.ndarray: Normalized array of audio features for each window
        """
        audio_wnd_arr: List[np.ndarray] = []
        start = 0
        self.audio_model.eval()

        with torch.no_grad():
            while (end := start + self.n_audio_frames) < waveform.shape[1]:
                wnd = waveform[:, start:end]
                features, _ = self.audio_model.extract_features(wnd)
                audio_feat = torch.max(features[-1], dim=1)[0].squeeze().cpu().numpy()
                audio_wnd_arr.append(audio_feat)
                start += self.n_audio_frames

        return (
            self.normalize_np(np.stack(audio_wnd_arr))
            if audio_wnd_arr
            else np.array([])
        )

    def get_spk_duration_feature(self, waveform: np.ndarray) -> np.ndarray:
        """
        Calculate speaker duration features from audio waveform.

        Computes the ratio of non-silent frames in each window.

        Args:
            waveform (np.ndarray): Audio waveform as numpy array

        Returns:
            np.ndarray: Array of speaker duration ratios for each window
        """
        audio_wnd_arr: List[float] = []
        start = 0

        while (end := start + self.n_audio_frames) < waveform.shape[1]:
            wnd = waveform[:, start:end]
            non_silence = wnd[:, ~(wnd == 0).all(0)]
            ratio = non_silence.shape[1] / wnd.shape[1]
            audio_wnd_arr.append(ratio)
            start += self.n_audio_frames

        return np.array(audio_wnd_arr)

    def preprocess(
        self, landmark_file_path: Path, audio_file_path: Path
    ) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        """
        Preprocess audio and landmark data.

        Loads and processes landmark data from a CSV file and audio data from a WAV file,
        removing silent frames and resampling audio to appropriate rates.

        Args:
            landmark_file_path (Path): Path to the landmark CSV file.
            audio_file_path (Path): Path to the audio WAV file.

        Returns:
            Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]: A tuple containing:
                - pd.DataFrame: Preprocessed landmark DataFrame with silent frames removed.
                - torch.Tensor: Normalized non-silent audio tensor (channels, time).
                - torch.Tensor: Full audio tensor resampled to target rate (channels, time).
        """
        # Load and preprocess audio waveform
        audio, audio_sample_rate = torchaudio.load(
            audio_file_path
        )  # (channel, timestamp)

        # Load and preprocess landmark data
        landmark_df = pd.read_csv(landmark_file_path)

        video_len = len(landmark_df)

        # Resample audio to video frame rate
        audio_in_video_fps = torchaudio.functional.resample(
            audio, audio_sample_rate, self.video_fps
        )
        audio_in_video_fps = audio_in_video_fps.squeeze().numpy()[:video_len]

        # Remove silent frames from landmark data
        silence_indices = np.where(audio_in_video_fps == 0)[0]
        landmark_df = landmark_df.drop(silence_indices)

        # Resample audio to target rate and normalize
        audio = torchaudio.functional.resample(audio, audio_sample_rate, self.audio_fps)
        normalized_audio = audio / (
            torch.max(torch.abs(audio)) + 1e-8
        )  # Avoid division by zero
        nonsilent_audio = normalized_audio[:, ~(normalized_audio == 0).all(0)]

        return landmark_df, nonsilent_audio, audio


def get_diarization_map(filename: Path) -> Dict[str, str]:
    """
    Load speaker diarization mapping from CSV file.

    Args:
        filename (Path): Path to diarization CSV file

    Returns:
        Dict[str, str]: Dictionary mapping filenames to speaker IDs
    """
    diarization_map = {}
    df = pd.read_csv(filename)

    for _, row in df.iterrows():
        diarization_map[row["Filename"]] = row["Speaker"]

    return diarization_map


def process_patient_data(
    patient: str,
    config: Dict[str, Any],
    extractor: FeatureExtractor,
    audio_dir: Path,
    landmark_dir: Path,
    diarization_map: Dict[str, str],
) -> None:
    """
    Process data for a single patient.

    Extracts features from audio, facial landmarks, and pose data
    and saves them as numpy arrays.

    Args:
        patient (str): Patient identifier
        config (Dict[str, Any]): Configuration dictionary
        extractor (FeatureExtractor): FeatureExtractor instance
        audio_dir (Path): Directory containing audio files
        landmark_dir (Path): Directory containing landmark files
        diarization_map (Dict[str, str]): Mapping from filenames to speaker IDs
    """

    features_dict: Dict[str, np.ndarray] = {}
    landmark_file_path = landmark_dir / f"{patient}.csv"
    audio_file_path = (
        audio_dir / patient / "audio" / f"{patient}_{diarization_map[patient]}.wav"
    )

    # Preprocess audio and landmark data
    landmark_df, nonsilent_audio, audio = extractor.preprocess(
        landmark_file_path, audio_file_path
    )

    # Extract features
    features_dict["face"] = extractor.get_facial_feature(landmark_df)
    features_dict["pose"] = extractor.get_pose_feature(landmark_df)
    features_dict["audio"] = extractor.get_audio_feature(
        nonsilent_audio.to(extractor.device)
    )
    features_dict["spk_duration"] = extractor.get_spk_duration_feature(audio.numpy())

    # Save features
    for modality, feature_np in features_dict.items():
        if feature_np.size == 0:
            logger.warning(f"Empty feature array for {patient}'s {modality}. Skipping.")
            continue

        embed_dir = Path(config["ensemble"]["path"]["embed_dir"]) / modality
        embed_dir.mkdir(parents=True, exist_ok=True)
        np.save(embed_dir / f"{patient}.npy", feature_np)


def encode() -> None:
    """
    Main function to encode features from all patients' data.

    Reads configuration, initializes the feature extractor, and processes
    data for all patients in the specified directories.

    Returns:
        None
    """
    # Load configuration
    config = read_config()

    # Ensure output directory exists
    Path(config["ensemble"]["path"]["embed_dir"]).mkdir(parents=True, exist_ok=True)

    # Initialize audio model
    audio_bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model = audio_bundle.get_model().to(device)
    logger.info(f"Using device: {device}")

    # Initialize feature extractor
    video_fps = 30
    resample_audio_fps = audio_bundle.sample_rate
    extractor = FeatureExtractor(
        config, audio_model, device, resample_audio_fps, video_fps
    )

    # Set up paths
    audio_dir = Path(config["ensemble"]["path"]["audio_diarized_dir"])
    landmark_dir = Path(config["ensemble"]["path"]["face_pose_landmark_dir"])

    # Validate directories
    for directory in [audio_dir, landmark_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

    # Get diarization map
    diarization_file = audio_dir / "diarization_results.csv"
    if not diarization_file.exists():
        raise FileNotFoundError(f"Diarization file not found: {diarization_file}")

    diarization_map = get_diarization_map(diarization_file)

    # Process each patient
    patient_count = 0
    for sub_dir in audio_dir.iterdir():
        if not sub_dir.is_dir():
            continue

        patient = sub_dir.stem
        logger.info(f"Processing: {patient}")
        try:
            process_patient_data(
                patient, config, extractor, audio_dir, landmark_dir, diarization_map
            )
            patient_count += 1
        except Exception as e:
            logger.error(f"Error processing {patient}: {e}")

    logger.info(f"Processing complete. Processed {patient_count} patients.")


if __name__ == "__main__":
    encode()
