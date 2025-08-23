"""
SchizoNegSympAI Inference Script

This script runs inference using trained models for assessing negative symptoms of schizophrenia.

Usage:
    # Run inference without labels (using default checkpoint directory)
    python inference.py

    # Run inference with specified experiment directory
    python inference.py --exp-dir /path/to/experiment/directory

    # Run inference with ground truth labels
    python inference.py --l
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config import read_config
from ensemble.estimator.trainer import SchizoTrainer

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SchizoNegSympAI Inference Script",
    )

    parser.add_argument(
        "--exp-dir", type=str, required=True, help="Path to experiment folder"
    )
    parser.add_argument(
        "--with-labels",
        "--l",
        action="store_true",
        default=False,
        help="Run inference with ground truth labels",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the inference pipeline.
    """
    config = read_config()
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory '{exp_dir}' does not exist.")

    analyzer = SchizoTrainer(config)

    for item in ["C10", "C11", "C12", "C13"]:
        analyzer.run_inference(item, exp_dir, with_labels=args.with_labels)


if __name__ == "__main__":
    main()
