"""
MediaPipe Holistic Tracking for Video Processing

This script performs holistic pose, face, and hand tracking on videos using MediaPipe.
It captures facial expressions, body posture, and hand movements, and outputs the data
to CSV files for further analysis.

Usage:
    # Process a single video file with default output directory
    python holistic_tracking.py /path/to/video.mp4

    # Process a single video file with custom output directory
    python holistic_tracking.py /path/to/video.mp4 --output-dir /path/to/output

    # Process a single video file with visualization enabled
    python holistic_tracking.py /path/to/video.mp4 --vis-dir /path/to/visualizations

    # Process all videos in a directory
    python holistic_tracking.py /path/to/video/directory
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import wget
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.config import read_config
from core.file_utils import is_video_file
from core.log import get_logger

config = read_config()
logger = get_logger()

# Blendshapes names used by MediaPipe face landmarker
BLENDSHAPES_NAME = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

CONFIDENCE_THRESH = 0.8  # Confidence threshold for detection


def download_face_landmarker(model_dir: Path) -> Path:
    """
    Download the face landmarker model if not already available.

    Args:
        model_dir (Path): Directory to save the model file.

    Returns:
        Path: Path to the downloaded model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    return Path(wget.download(model_url, str(model_dir)))


def get_face_landmarker() -> vision.FaceLandmarker:
    """
    Initialize and return the MediaPipe FaceLandmarker with appropriate configurations.

    Returns:
        vision.FaceLandmarker: Configured face landmarker object
    """
    # Face landmarker configuration
    model_dir = Path(__file__).parent / ".cache" / "mediapipe"
    model_path = model_dir / "face_landmarker.task"
    if not model_path.exists():
        model_path = download_face_landmarker(model_dir)

    base_options = python.BaseOptions(model_asset_path=str(model_path))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=CONFIDENCE_THRESH,
        min_face_presence_confidence=CONFIDENCE_THRESH,
        min_tracking_confidence=CONFIDENCE_THRESH,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    return face_landmarker


def holistic_tracking_on_video(
    vid_file: Path,
    output_csv: Path,
    visualize_dir: Optional[Path] = None,
) -> None:
    """
    Process a video file with MediaPipe holistic tracking.

    This function extracts face landmarks, blendshapes, body pose, and hand positions
    from each frame of the video, and saves the data to a CSV file.

    Args:
        vid_file (Path): Path to the input video file
        output_csv (Path): Path to save the output CSV file
        visualize_dir (Optional[Path]): Optional path to save visualization images
    """
    vid_name = vid_file.stem

    cap = cv2.VideoCapture(str(vid_file))

    timestamp = 0
    output_df = pd.DataFrame()

    face_landmarker = get_face_landmarker()

    # Holistic landmark detection
    with mp.solutions.holistic.Holistic(
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=CONFIDENCE_THRESH,
        min_tracking_confidence=CONFIDENCE_THRESH,
    ) as holistic:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            timestamp += 1
            logger.info(f"Processing video: {vid_file.name} | Frame: {timestamp}")

            # Initialize result dictionary for this frame
            output_results = initialize_output_results(timestamp)

            # Convert the BGR image to RGB before processing
            holistic_results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Process pose landmarks if detected
            if holistic_results.pose_world_landmarks:
                output_results["poseDetected"] = [1]
                process_pose_landmarks(output_results, holistic_results)

            # Process left hand landmarks if detected
            if holistic_results.left_hand_landmarks:
                output_results["leftHandDetected"] = [1]
                process_hand_landmarks(
                    output_results, holistic_results.left_hand_landmarks, "leftHand"
                )

            # Process right hand landmarks if detected
            if holistic_results.right_hand_landmarks:
                output_results["rightHandDetected"] = [1]
                process_hand_landmarks(
                    output_results, holistic_results.right_hand_landmarks, "rightHand"
                )

            # Generate visualization if requested
            if visualize_dir is not None:
                visualize_holistic_results(
                    visualize_dir, vid_name, timestamp, image, holistic_results
                )

            # Process face landmarks if detected
            if holistic_results.face_landmarks:
                face_result, face_image = process_face_landmarks(
                    image, holistic_results, face_landmarker, output_results
                )

                # Generate visualization if requested
                if visualize_dir is not None and face_result.face_landmarks:
                    visualize_face_result(
                        visualize_dir, vid_name, timestamp, face_image, face_result
                    )

            # Add frame data to dataframe
            new_row = pd.DataFrame.from_dict(output_results)
            output_df = pd.concat([output_df, new_row], ignore_index=True, axis=0)

    cap.release()
    output_df.to_csv(output_csv, index=False)


def initialize_output_results(timestamp: int) -> Dict[str, List]:
    """
    Initialize the output results dictionary for a frame.

    Args:
        timestamp (int): The current frame number

    Returns:
        Dict[str, List]: Dictionary with initialized values for all tracked landmarks
    """
    output_results = {
        "timestamp": [timestamp],
        "faceDetected": [0],
        "poseDetected": [0],
        "leftHandDetected": [0],
        "rightHandDetected": [0],
    }

    # Initialize blendshapes
    for i in range(len(BLENDSHAPES_NAME)):
        output_results[f"blendShapes.{BLENDSHAPES_NAME[i]}"] = [0.0]

    # Initialize face landmarks
    for i in range(478):
        output_results[f"face.{i}.x"] = [0.0]
        output_results[f"face.{i}.y"] = [0.0]
        output_results[f"face.{i}.z"] = [0.0]

    # Initialize pose landmarks
    for i in range(33):
        output_results[f"pose.{i}.x"] = [0.0]
        output_results[f"pose.{i}.y"] = [0.0]
        output_results[f"pose.{i}.z"] = [0.0]
        output_results[f"pose.{i}.visibility"] = [0.0]

    # Initialize hand landmarks
    for hand in ["leftHand", "rightHand"]:
        for i in range(21):
            output_results[f"{hand}.{i}.x"] = [0.0]
            output_results[f"{hand}.{i}.y"] = [0.0]
            output_results[f"{hand}.{i}.z"] = [0.0]

    return output_results


def process_pose_landmarks(output_results: Dict[str, List], holistic_results) -> None:
    """
    Process pose landmarks and add them to the output results.

    Args:
        output_results (Dict[str, List]): Dictionary to store the results
        holistic_results: Results from MediaPipe holistic detection
    """
    for i, lmk in enumerate(holistic_results.pose_world_landmarks.landmark):
        output_results[f"pose.{i}.x"] = [lmk.x]
        output_results[f"pose.{i}.y"] = [lmk.y]
        output_results[f"pose.{i}.z"] = [lmk.z]
        output_results[f"pose.{i}.visibility"] = [lmk.visibility]


def process_hand_landmarks(
    output_results: Dict[str, List], hand_landmarks, hand_type: str
) -> None:
    """
    Process hand landmarks and add them to the output results.

    Args:
        output_results: Dictionary to store the results
        hand_landmarks: Hand landmarks from MediaPipe detection
        hand_type: Either "leftHand" or "rightHand"
    """
    for i, lmk in enumerate(hand_landmarks.landmark):
        output_results[f"{hand_type}.{i}.x"] = [lmk.x]
        output_results[f"{hand_type}.{i}.y"] = [lmk.y]
        output_results[f"{hand_type}.{i}.z"] = [lmk.z]


def process_face_landmarks(
    image: np.ndarray,
    holistic_results,
    face_landmarker: vision.FaceLandmarker,
    output_results: Dict[str, List],
) -> Tuple[vision.FaceLandmarkerResult, np.ndarray]:
    """
    Process face landmarks and blendshapes.

    Args:
        image (np.ndarray): Input image frame
        holistic_results: Results from MediaPipe holistic detection
        face_landmarker (vision.FaceLandmarker): MediaPipe face landmarker object
        output_results (Dict[str, List]): Dictionary to store the results

    Returns:
        Tuple[vision.FaceLandmarkerResult, np.ndarray]: Face landmark detection result and face image
    """
    cropped_face_image = crop_face(image, holistic_results)

    face_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.array(cropped_face_image),
    )
    face_result = face_landmarker.detect(face_image)

    # Face is detected by holistic tracking, but may not be detected in cropped image
    if face_result.face_landmarks:
        output_results["faceDetected"] = [1]

        for i, lmk in enumerate(face_result.face_landmarks[0]):
            output_results[f"face.{i}.x"] = [lmk.x]
            output_results[f"face.{i}.y"] = [lmk.y]
            output_results[f"face.{i}.z"] = [lmk.z]

        for bs in face_result.face_blendshapes[0]:
            output_results[f"blendShapes.{bs.category_name}"] = [bs.score]

    return face_result, face_image.numpy_view()


def visualize_holistic_results(
    visualize_dir: Path,
    vid_name: str,
    timestamp: int,
    image: np.ndarray,
    holistic_results,
) -> None:
    """
    Generate visualization of holistic tracking results.

    Args:
        visualize_dir (Path): Path to save visualization images
        vid_name (str): Video name for folder structure
        timestamp (int): Current frame number
        image (np.ndarray): Input image frame
        holistic_results: Results from MediaPipe holistic detection
    """
    holistic_visualize_dir = visualize_dir / vid_name / "holistic"
    holistic_visualize_dir.mkdir(parents=True, exist_ok=True)

    annotated_holistic_image = draw_holistic_landmarks_on_image(image, holistic_results)
    cv2.imwrite(
        str(holistic_visualize_dir / f"{str(timestamp).zfill(6)}.png"),
        annotated_holistic_image,
    )


def visualize_face_result(
    visualize_dir: Path,
    vid_name: str,
    timestamp: int,
    face_image: np.ndarray,
    face_result: vision.FaceLandmarkerResult,
) -> None:
    """
    Generate visualizations for face landmarks and blendshapes.

    Args:
        visualize_dir (Path): Path to save visualization images
        vid_name (str): Video name for folder structure
        timestamp (int): Current frame number
        face_image (np.ndarray): Face image to annotate
        face_result (vision.FaceLandmarkerResult): FaceLandmarker detection result
    """
    face_visualize_dir = visualize_dir / vid_name / "face"
    face_visualize_dir.mkdir(parents=True, exist_ok=True)
    annotated_face_image = draw_face_landmarks_on_image(face_image, face_result)
    cv2.imwrite(
        str(face_visualize_dir / f"{str(timestamp).zfill(6)}.png"),
        annotated_face_image,
    )

    bshape_visualize_dir = visualize_dir / vid_name / "bshape"
    bshape_visualize_dir.mkdir(parents=True, exist_ok=True)
    blendshapes_bar = plot_face_blendshapes_bar_graph(face_result.face_blendshapes[0])
    blendshapes_bar.savefig(
        str(bshape_visualize_dir / f"{str(timestamp).zfill(6)}.png"),
        dpi=300,
        bbox_inches="tight",
    )


def draw_face_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: vision.FaceLandmarkerResult
) -> np.ndarray:
    """
    Draws face landmarks on the input image.

    Args:
        rgb_image (np.ndarray): The RGB image on which to draw landmarks
        detection_result (vision.FaceLandmarkerResult): The result of MediaPipe face landmark detection

    Returns:
        np.ndarray: Annotated image with face landmarks
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        # Draw the face landmarks
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes) -> plt.Figure:
    """
    Plots a bar graph of face blendshapes.

    Args:
        face_blendshapes: A list of face blendshapes data

    Returns:
        plt.Figure: The bar graph figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlabel("Value")
    ax.set_title("Face Blendshapes")

    if face_blendshapes is not None:
        # Extract the face blendshapes category names and scores
        face_blendshapes_names = [
            face_blendshapes_category.category_name
            for face_blendshapes_category in face_blendshapes
        ]
        face_blendshapes_scores = [
            face_blendshapes_category.score
            for face_blendshapes_category in face_blendshapes
        ]
        # The blendshapes are ordered in decreasing score value
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        bar = ax.barh(
            face_blendshapes_ranks,
            face_blendshapes_scores,
            label=[str(x) for x in face_blendshapes_ranks],
        )
        ax.set_yticks(face_blendshapes_ranks)
        ax.set_yticklabels(face_blendshapes_names, fontsize=10)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            ax.text(
                patch.get_x()
                + patch.get_width()
                + 0.001,  # Move the text slightly to the right
                patch.get_y() + patch.get_height() / 2,
                f"{score:.3f}",
                va="center",
            )

    plt.close(fig)
    return fig


def draw_holistic_landmarks_on_image(
    rgb_image: np.ndarray, detection_result
) -> np.ndarray:
    """
    Draws holistic landmarks on the input image.

    Args:
        rgb_image (np.ndarray): The RGB image on which to draw landmarks
        detection_result: The result of MediaPipe holistic landmark detection

    Returns:
        np.ndarray: Annotated image with holistic landmarks
    """
    BG_COLOR = (192, 192, 192)  # gray
    annotated_image = rgb_image.copy()

    # Draw segmentation on the image
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "holistic_results.segmentation_mask" with "image"
    if detection_result.segmentation_mask is not None:
        condition = np.stack((detection_result.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(rgb_image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

    # Draw face landmarks on the image
    if detection_result.face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            detection_result.face_landmarks,
            mp.solutions.holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    # Draw pose landmarks on the image
    if detection_result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            detection_result.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    # Draw hand landmarks (missing in original code)
    if detection_result.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            detection_result.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        )
    if detection_result.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            detection_result.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        )

    return annotated_image


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Optional[Tuple[int, int]]:
    """
    Converts normalized value pair to pixel coordinates.

    Args:
        normalized_x (float): Normalized x coordinate (0.0 to 1.0)
        normalized_y (float): Normalized y coordinate (0.0 to 1.0)
        image_width (int): Width of the image
        image_height (int): Height of the image

    Returns:
        Optional[Tuple[int, int]]: Tuple of (x_pixel, y_pixel) or None if coordinates are invalid
    """

    # Checks if the float value is between 0 and 1
    def is_valid_normalized_value(value: float) -> bool:
        return 0 <= value <= 1

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def crop_face(image: np.ndarray, holistic_results) -> np.ndarray:
    """
    Crop the face region from the input image based on holistic landmarks.

    Args:
        image (np.ndarray): The input image
        holistic_results: The result of MediaPipe holistic landmark detection

    Returns:
        np.ndarray: The cropped face image
    """
    SCALE_X = 1.5  # Scale factor for width of cropped face
    SCALE_Y = 1.5  # Scale factor for height of cropped face
    image_height, image_width, _ = image.shape

    # Extract normalized coordinates
    normalized_x_coordinates = []
    normalized_y_coordinates = []
    for lmk in holistic_results.face_landmarks.landmark:
        normalized_x_coordinates.append(lmk.x)
        normalized_y_coordinates.append(lmk.y)

    # Find boundaries of face with clamping to image edges
    min_normalized_x = max(min(normalized_x_coordinates), 0)
    min_normalized_y = max(min(normalized_y_coordinates), 0)
    max_normalized_x = min(max(normalized_x_coordinates), 1)
    max_normalized_y = min(max(normalized_y_coordinates), 1)

    # Convert to pixel coordinates
    x1, y1 = normalized_to_pixel_coordinates(
        min_normalized_x, min_normalized_y, image_width, image_height
    )
    x2, y2 = normalized_to_pixel_coordinates(
        max_normalized_x, max_normalized_y, image_width, image_height
    )

    # Apply scaling centered on face
    center_x = round((x1 + x2) / 2)
    center_y = round((y1 + y2) / 2)
    half_width = round(((x2 - x1) * SCALE_X) / 2)
    half_height = round(((y2 - y1) * SCALE_Y) / 2)

    # Calculate scaled crop boundaries with clamping to image edges
    scaled_x1 = max(center_x - half_width, 0)
    scaled_y1 = max(center_y - half_height, 0)
    scaled_x2 = min(center_x + half_width, image_width - 1)
    scaled_y2 = min(center_y + half_height, image_height - 1)

    # Extract cropped region
    cropped_image = image[scaled_y1:scaled_y2, scaled_x1:scaled_x2]

    return cropped_image


def run_single(
    input_path: Path,
    output_dir: Path,
    visualize_dir: Optional[Path] = None,
) -> None:
    """
    Process a single video file.

    Args:
        input_path (Path): Path to the input video file
        output_dir (Path): Path to save output data
        visualize_dir (Optional[Path]): Optional path to save visualization images
    """
    output_csv = output_dir / f"{input_path.stem}.csv"

    if not is_video_file(input_path):
        raise ValueError(f"'{input_path}' is not a valid video file")

    holistic_tracking_on_video(
        input_path,
        output_csv,
        visualize_dir=visualize_dir,
    )


def run_folder(
    input_dir: Path,
    output_dir: Path,
    visualize_dir: Optional[Path] = None,
) -> None:
    """
    Process all video files in a directory.

    Args:
        input_dir (Path): Path to directory containing input video files
        output_dir (Path): Path to save output data
        visualize_dir (Optional[Path]): Optional path to save visualization images
    """
    for file_path in sorted(input_dir.glob("*")):
        if not is_video_file(file_path):
            logger.info(f"Skipping non-video file: {file_path}")
            continue

        output_csv = output_dir / f"{file_path.stem}.csv"

        # Skip if already processed
        if output_csv.exists():
            logger.info(f"Skipping already processed file: {file_path}")
            continue

        holistic_tracking_on_video(
            file_path,
            output_csv,
            visualize_dir=visualize_dir,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MediaPipe Holistic Tracking for Video Processing"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file or directory containing videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config["ensemble"]["path"]["face_pose_landmark_dir"],
        help="Directory to store processed output",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Output folder for visualization images",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the holistic tracking workflow.
    """
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    visualize_dir = Path(args.vis_dir) if args.vis_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path '{args.input}' does not exist")

    output_dir.mkdir(parents=True, exist_ok=True)
    if visualize_dir:
        visualize_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        run_single(
            input_path,
            output_dir,
            visualize_dir=visualize_dir,
        )
    elif input_path.is_dir():
        run_folder(
            input_path,
            output_dir,
            visualize_dir=visualize_dir,
        )
    else:
        raise ValueError(f"Input path '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()
