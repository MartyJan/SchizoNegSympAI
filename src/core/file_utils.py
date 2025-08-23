from pathlib import Path
from typing import Any, Dict

import yaml


def is_file_of_type(file_path: Path, extensions: tuple) -> bool:
    """
    Check if a file is of a specific type based on its extension.

    Args:
        file_path (Path): Path of the file to check.
        extensions (tuple): Tuple of valid file extensions.

    Returns:
        bool: True if the file matches the extensions, False otherwise.
    """
    return file_path.is_file() and file_path.suffix.lower() in extensions


def is_audio_file(file_path: Path) -> bool:
    """
    Check if a file is an audio file based on its extension.

    Args:
        file_path (Path): Path of the file to check.

    Returns:
        bool: True if the file is an audio file, False otherwise.
    """
    audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")
    return is_file_of_type(file_path, audio_extensions)


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image file based on its extension.

    Args:
        file_path (Path): Path of the file to check.

    Returns:
        bool: True if the file is an image file, False otherwise.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    return is_file_of_type(file_path, image_extensions)


def is_video_file(file_path: Path) -> bool:
    """
    Check if a file is a video file based on its extension.

    Args:
        file_path (Path): Path of the file to check.

    Returns:
        bool: True if the file is a video file, False otherwise.
    """
    video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv")
    return is_file_of_type(file_path, video_extensions)


def convert_dict_to_yaml(dictionary: Dict[str, Any], file_path: Path) -> None:
    """
    Convert a Python dictionary to YAML format and save it to a file.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be converted to YAML.
        file_path (Path): Path object where the YAML content will be saved.

    Returns:
        None: This function does not return any value.
    """
    with file_path.open("w") as f:
        yaml.dump(dictionary, f, default_flow_style=False, sort_keys=False)
