"""
SchizoNegSympAI Training Script

This script trains ensemble estimator models for assessing negative symptoms of schizophrenia.

Usage:
    python train.py
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.config import read_config
from ensemble.estimator.trainer import SchizoTrainer

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def main() -> None:
    """
    Main function to execute the training pipeline.
    """
    config = read_config()
    analyzer = SchizoTrainer(config)

    for item in ["C10", "C11", "C12", "C13"]:
        analyzer.run_train(item)


if __name__ == "__main__":
    main()
