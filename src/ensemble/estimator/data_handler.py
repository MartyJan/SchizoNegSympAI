import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel


class Patient(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """
    Class representing a patient's data.

    Args:
        name (str): The name of the patient.
        label (int): The label associated with the patient.
        sex (str): The sex of the patient.
        raw_features (Dict[str, np.ndarray]): Dictionary mapping modality names to feature arrays.
        model_input (Optional[np.ndarray]): Processed features ready for model input. Defaults to None.
    """

    name: str
    label: int
    sex: str
    raw_features: Dict[str, np.ndarray]
    model_input: Optional[np.ndarray] = None


class PatientDataHandler:
    """
    Handles loading, processing, and organizing patient data for analysis.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        require_labels: bool = True,
    ) -> None:
        """
        Initialize the PatientDataHandler with configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
            logger (logging.Logger): Logger instance for logging messages.
            require_labels (bool): Flag indicating whether to require labels.
        """
        self.config = config
        self.logger = logger
        self.labels = None
        if require_labels:
            self.labels = pd.read_csv(
                Path(self.config["ensemble"]["path"]["label_file"])
            )

    def get_patient_data(
        self, assessment_item: Union[str, int], modalities: List[str]
    ) -> List[Patient]:
        """
        Read patient data based on available files in modality folders and create a list of Patient objects.
        First finds all patients with complete data across modalities, then retrieves their labels.

        Args:
            assessment_item (Union[str, int]): The assessment item/column to use as label.
            modalities (List[str]): List of modalities to include.

        Returns:
            List[Patient]: A list of Patient objects with data from all specified modalities.
        """
        embed_dir = Path(self.config["ensemble"]["path"]["embed_dir"])
        labels_df = self.labels

        # First, find all patients with data for all specified modalities
        available_patients: Set[str] = set()
        first_modality = True

        # For each modality, get set of available patient names
        for modality in modalities:
            modality_dir = embed_dir / modality
            if not modality_dir.exists():
                msg = f"Directory for modality '{modality}' not found at {modality_dir}"
                self.logger.warning(msg)
                continue

            # Get all patient names from this modality directory
            patient_names = {p.stem for p in modality_dir.glob("*.npy")}

            # For first modality, initialize the set
            if first_modality:
                available_patients = patient_names
                first_modality = False
            else:
                # Keep only patients that have data for all modalities
                available_patients &= patient_names

        if not available_patients:
            raise ValueError(
                "No patients with data available for all requested modalities"
            )

        def get_info_from_labels(
            df: pd.DataFrame, patient_name: str
        ) -> Optional[Tuple[int, str]]:
            """
            Extract label and sex information for a patient.

            Args:
                df (pd.DataFrame): DataFrame containing patient labels.
                patient_name (str): Name of the patient to extract information for.

            Returns:
                Optional[Tuple[int, str]]: Tuple containing label and sex if found, None otherwise.
            """
            gt = df[df["Record ID"] == patient_name]
            if gt.empty:
                return None

            column_name = assessment_item
            if isinstance(column_name, int):
                trueY = gt.iloc[:, column_name].item()
            else:
                trueY = gt[column_name].item()

            if not isinstance(trueY, int) or trueY < 0:
                return None

            sex = gt["Sex"].values[0] if "Sex" in gt.columns else "unknown"
            return trueY, sex

        # Process each available patient
        patients = []
        for name in sorted(available_patients):
            if labels_df is None:
                label, sex = -1, "unknown"
            else:
                res = get_info_from_labels(labels_df, name)
                if res is None:
                    msg = f"Missing label for patient {name}"
                    self.logger.warning(msg)
                    continue
                else:
                    label, sex = res

            # Load features for all modalities
            features = {}
            for modality in modalities:
                embed_path = embed_dir / modality / f"{name}.npy"
                features[modality] = np.load(embed_path)

            patients.append(
                Patient(name=name, label=label, sex=sex, raw_features=features)
            )

        return patients
