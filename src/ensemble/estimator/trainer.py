import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from third_party.OrdClass import OrdClass
from third_party.SpectralPool import SpectralPool2d

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.eval import evaluate_metrics
from core.file_utils import convert_dict_to_yaml
from core.log import get_logger
from ensemble.estimator.data_handler import Patient, PatientDataHandler


class SchizoTrainer:
    """
    Class for handling schizophrenia negative symptom estimation models, including data processing,
    model training, and evaluation for different assessment items.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SchizoTrainer class.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing training parameters,
                                    file paths, and other settings.
        """
        self.config = config

        # Set up experiment name and log folder
        now = datetime.now().astimezone()  # Get local timezone from machine
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        # Set log directory based on provided path or generate from config with timestamp

        experiment_name = config["ensemble"]["training"]["experiment_name"]
        result_dir = Path(config["ensemble"]["path"]["result_dir"])
        self.log_dir = result_dir / f"{experiment_name}_{timestamp}"

        self.logger = get_logger(self.log_dir)

    def get_model_input(
        self, patients: List[Patient], modality: List[str], pooling_dimension: List[int]
    ) -> List[Patient]:
        """
        Process patient data to create model inputs using spectral pooling.

        Args:
            patients (List[Patient]): List of patient objects to process.
            modality (List[str]): List of modalities to use for feature extraction.
            pooling_dimension (List[int]): Dimensions to use for spectral pooling.

        Returns:
            List[Patient]: List of patient objects with model_input field updated with pooled features.
        """
        for patient in patients:
            used_features = {}
            for m in modality:
                used_features[m] = torch.from_numpy(
                    patient.raw_features[m].astype("float32")
                )

            pool_feature_list = []
            for _, feature in used_features.items():
                # (timestamp, features) -> (features, timestamp) -> (1, 1, features, timestamp)
                if feature.ndim == 1:
                    x = feature.view(1, -1).unsqueeze(0).unsqueeze(0)
                else:
                    x = feature.permute(1, 0).unsqueeze(0).unsqueeze(0)

                oheight = x.size(-2)
                for owidth in pooling_dimension:
                    pooled = self._spectral_pool(x, oheight, owidth)
                    pool_feature_list.append(pooled.numpy().flatten())

            patient.model_input = np.hstack(pool_feature_list)

        return patients

    def _spectral_pool(
        self, x: torch.Tensor, oheight: int, owidth: int
    ) -> torch.Tensor:
        """
        Apply spectral pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor for pooling with shape (batch, channels, height, width).
            oheight (int): Output height after pooling.
            owidth (int): Output width after pooling.

        Returns:
            torch.Tensor: Pooled tensor with reduced dimensions.
        """
        pooling = SpectralPool2d(oheight=oheight, owidth=owidth)
        pooled = pooling(x)

        # Since the second dimension is all zeros when feature is 1D, only take the first dimension
        if oheight == 1:
            pooled = pooled[:, :, 0, :]

        return pooled

    def train(self, train_patients: List[Patient]) -> Dict[str, GridSearchCV]:
        """
        Train ordinal and nominal Random Forest models on the given patient data.

        Args:
            train_patients (List[Patient]): List of patient objects with model_input data for training.

        Returns:
            Dict[str, GridSearchCV]: Dictionary containing trained ordinal Random Forest model
                                    ('ord_rf') and nominal Random Forest model ('nom_rf').
        """

        trainX = np.array([p.model_input for p in train_patients])
        trainY = np.array([p.label for p in train_patients])

        class_counts = Counter(trainY)
        min_class_count = min(class_counts.values())

        if len(class_counts) < 2:
            raise ValueError(
                "Not enough classes for training. At least two classes are required."
            )

        # Convert NaN to number
        trainX = np.nan_to_num(trainX)

        # Apply downsampling if configured
        if self.config["ensemble"]["training"]["downsample"] == "TomekLinks":
            usm = TomekLinks(sampling_strategy=[0])
            trainX, trainY = usm.fit_resample(trainX, trainY)

        models = {}

        # Train both model architectures
        for archit in ["ord_rf", "nom_rf"]:
            # Initialize classifier
            if archit == "ord_rf":
                clf = OrdClass(
                    classifier=RandomForestClassifier(
                        random_state=self.config["ensemble"]["training"]["random_seed"]
                    )
                )
                param_grid = {
                    "clf__classifier__n_estimators": self.config["ensemble"][
                        "training"
                    ]["classifier"]["n_estimators"],
                    "clf__classifier__max_depth": self.config["ensemble"]["training"][
                        "classifier"
                    ]["max_depth"],
                }
            else:  # archit == "nom_rf"
                clf = RandomForestClassifier(
                    random_state=self.config["ensemble"]["training"]["random_seed"]
                )
                param_grid = {
                    "clf__n_estimators": self.config["ensemble"]["training"][
                        "classifier"
                    ]["n_estimators"],
                    "clf__max_depth": self.config["ensemble"]["training"]["classifier"][
                        "max_depth"
                    ],
                }

            # Create pipeline with feature selection
            pipe = Pipeline(
                [
                    (
                        "fs",
                        SelectFromModel(
                            RandomForestClassifier(
                                n_estimators=self.config["ensemble"]["training"][
                                    "feature_selector"
                                ]["n_estimators"],
                                max_depth=self.config["ensemble"]["training"][
                                    "feature_selector"
                                ]["max_depth"],
                                random_state=self.config["ensemble"]["training"][
                                    "random_seed"
                                ],
                            )
                        ),
                    ),
                    ("clf", clf),
                ]
            )

            # Only use GridSearchCV if enough samples per class
            if min_class_count >= 5:
                search = GridSearchCV(pipe, param_grid, scoring="balanced_accuracy")
                model = search.fit(X=trainX, y=trainY)
            else:
                self.logger.warning(
                    f"Not enough samples per class for GridSearchCV (min_class_count={min_class_count}). "
                    "Fitting pipeline with default parameters instead."
                )
                fitted_pipe = pipe.fit(trainX, trainY)
                model = SimpleNamespace(
                    best_estimator_=fitted_pipe,
                    best_params_={},
                    predict_proba=fitted_pipe.predict_proba,
                    predict=fitted_pipe.predict,
                )

            models[archit] = model

        return models

    def test(
        self,
        test_patients: List[Patient],
        ord_rf_model: GridSearchCV,
        nom_rf_model: GridSearchCV,
    ) -> Dict[str, Dict[str, Union[np.ndarray, List[str]]]]:
        """
        Test models on test data and generate estimations, including an ensemble approach.

        Args:
            test_patients (List[Patient]): List of patient objects with model_input data for testing.
            ord_rf_model (GridSearchCV): Trained Ordinal Random Forest model.
            nom_rf_model (GridSearchCV): Trained Nominal Random Forest model.

        Returns:
            Dict[str, Dict[str, Union[np.ndarray, List[str]]]]: Dictionary containing estimation results for each model
            with keys 'ord_rf', 'nom_rf', and 'ensemble'. Each model entry contains:
            - 'truth': Ground truth labels
            - 'pred': Predicted labels
            - 'patient_names': List of patient identifiers
        """
        # Pad model_input arrays to the same length
        model_inputs = [p.model_input for p in test_patients]
        max_len = max(arr.shape[0] for arr in model_inputs)
        testX = np.stack(
            [
                np.pad(arr, (0, max_len - arr.shape[0]), mode="constant")
                for arr in model_inputs
            ]
        )

        testY = np.array([p.label for p in test_patients])
        test_patient_names = [p.name for p in test_patients]

        # Convert NaN to number
        testX = np.nan_to_num(testX)

        pred_arrays = {}

        # Check and pad testX if needed for ord_rf_model
        ord_rf_input_dim = ord_rf_model.best_estimator_.named_steps[
            "fs"
        ].estimator_.n_features_in_
        if testX.shape[1] < ord_rf_input_dim:
            pad_width = ord_rf_input_dim - testX.shape[1]
            testX_ord = np.pad(testX, ((0, 0), (0, pad_width)), mode="constant")
        elif testX.shape[1] > ord_rf_input_dim:
            testX_ord = testX[:, :ord_rf_input_dim]
        else:
            testX_ord = testX

        # Test Ordinal RF model
        ord_rf_predY_prob = ord_rf_model.predict_proba(testX_ord)
        ord_rf_predY = np.argmax(ord_rf_predY_prob, axis=1)
        pred_arrays["ord_rf"] = {
            "truth": testY,
            "pred": ord_rf_predY,
            "patient_names": test_patient_names,
        }

        # Check and pad testX if needed for nom_rf_model
        nom_rf_input_dim = nom_rf_model.best_estimator_.named_steps[
            "fs"
        ].estimator_.n_features_in_
        if testX.shape[1] < nom_rf_input_dim:
            pad_width = nom_rf_input_dim - testX.shape[1]
            testX_nom = np.pad(testX, ((0, 0), (0, pad_width)), mode="constant")
        elif testX.shape[1] > nom_rf_input_dim:
            testX_nom = testX[:, :nom_rf_input_dim]
        else:
            testX_nom = testX

        # Test RF model
        nom_rf_predY_prob = nom_rf_model.predict_proba(testX_nom)
        nom_rf_predY = np.argmax(nom_rf_predY_prob, axis=1)
        pred_arrays["nom_rf"] = {
            "truth": testY,
            "pred": nom_rf_predY,
            "patient_names": test_patient_names,
        }

        # Ensemble based on confidence
        ord_rf_prob_max = np.max(ord_rf_predY_prob, axis=1)
        nom_rf_prob_max = np.max(nom_rf_predY_prob, axis=1)

        ord_rf_indices = np.where(ord_rf_prob_max >= nom_rf_prob_max)[0]
        nom_rf_indices = np.array(
            list(set(range(len(ord_rf_predY_prob))) - set(ord_rf_indices))
        )

        if len(ord_rf_indices) > 0 and len(nom_rf_indices) > 0:
            ensemble_testY = np.concatenate(
                (testY[ord_rf_indices], testY[nom_rf_indices])
            )
            ensemble_predY = np.concatenate(
                (
                    np.argmax(ord_rf_predY_prob[ord_rf_indices], axis=1),
                    np.argmax(nom_rf_predY_prob[nom_rf_indices], axis=1),
                )
            )
            ensemble_patient_names = [
                test_patient_names[i]
                for i in list(ord_rf_indices) + list(nom_rf_indices)
            ]
        elif len(ord_rf_indices) > 0:
            ensemble_testY = testY[ord_rf_indices]
            ensemble_predY = np.argmax(ord_rf_predY_prob[ord_rf_indices], axis=1)
            ensemble_patient_names = [test_patient_names[i] for i in ord_rf_indices]
        elif len(nom_rf_indices) > 0:
            ensemble_testY = testY[nom_rf_indices]
            ensemble_predY = np.argmax(nom_rf_predY_prob[nom_rf_indices], axis=1)
            ensemble_patient_names = [test_patient_names[i] for i in nom_rf_indices]
        else:
            # Fallback if no indices are selected
            ensemble_testY = testY
            ensemble_predY = nom_rf_predY
            ensemble_patient_names = test_patient_names

        pred_arrays["ensemble"] = {
            "truth": ensemble_testY,
            "pred": ensemble_predY,
            "patient_names": ensemble_patient_names,
        }

        return pred_arrays

    def create_prediction_df(
        self,
        patient_names: List[str],
        predictions: Optional[np.ndarray],
        ground_truth: Optional[np.ndarray],
        column_name: str,
    ) -> pd.DataFrame:
        """
        Create a DataFrame containing predictions and ground truth values for each patient.

        Args:
            patient_names (List[str]): List of patient identifiers.
            predictions (Optional[np.ndarray]): Predicted labels or None.
            ground_truth (Optional[np.ndarray]): Ground truth labels or None.
            column_name (str): Name of the assessment column for predictions.

        Returns:
            pd.DataFrame: DataFrame with columns ['Patient', 'Rater', column_name], containing predictions
                          (with Rater='Machine'), ground truth (with Rater='Doctor'), or both if available.
        """

        # Predictions
        pred_df = None
        if predictions is not None:
            pred_df = pd.DataFrame(
                {
                    "Patient": patient_names,
                    "Rater": ["Machine"] * len(patient_names),
                    column_name: predictions,
                }
            )

        # Ground truth
        label_df = None
        if ground_truth is not None:
            label_df = pd.DataFrame(
                {
                    "Patient": patient_names,
                    "Rater": ["Doctor"] * len(patient_names),
                    column_name: ground_truth,
                }
            )

        # Combine
        if pred_df is not None and label_df is not None:
            return pd.concat([pred_df, label_df])
        elif pred_df is not None:
            return pred_df
        elif label_df is not None:
            return label_df
        else:
            return pd.DataFrame()

    def run_train(self, assessment_item: str) -> None:
        """
        Run the training process for a specific assessment item using all available data.

        This method trains models on the entire dataset for a given assessment item and
        saves the trained models for later use in inference.

        Args:
            assessment_item (str): The assessment item/column to use as the estimation target.
        """

        data_handler = PatientDataHandler(self.config, self.logger)

        # Load patients
        all_patients = data_handler.get_patient_data(
            assessment_item,
            self.config["ensemble"]["training"]["modality"][assessment_item],
        )
        self.logger.info(
            f"[ Train | {assessment_item} ] Total patients: {len(all_patients)}"
        )
        self.logger.info(
            f"[ Train | {assessment_item} ] Patients: {[p.name for p in all_patients]}"
        )

        train_log_dir = self.log_dir.with_name(f"train_{self.log_dir.name}")
        train_log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        convert_dict_to_yaml(self.config, train_log_dir / "config.yaml")

        train_patients = self.get_model_input(
            all_patients,
            self.config["ensemble"]["training"]["modality"][assessment_item],
            self.config["ensemble"]["training"]["pooling_dimension"][assessment_item],
        )

        # Train models
        models = self.train(train_patients)
        ord_rf_model = models["ord_rf"]
        nom_rf_model = models["nom_rf"]

        self.logger.info(
            f"[ Train | {assessment_item} ] ord_rf best params: {ord_rf_model.best_params_}"
        )
        self.logger.info(
            f"[ Train | {assessment_item} ] nom_rf best params: {nom_rf_model.best_params_}"
        )

        # Save model checkpoints
        checkpoint_dir = train_log_dir / assessment_item / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save both models using pickle
        with open(checkpoint_dir / "ord_rf_model.pkl", "wb") as f:
            pickle.dump(ord_rf_model, f)
        with open(checkpoint_dir / "nom_rf_model.pkl", "wb") as f:
            pickle.dump(nom_rf_model, f)

        self.logger.info(f"[ Train | {assessment_item} ] Models saved successfully")

    def run_inference(
        self, assessment_item: str, experiment_dir: Path, with_labels: bool = False
    ) -> None:
        """
        Run inference using previously trained models on patient data.

        This method loads trained models and performs negative symptom assessment on patient data.
        It can optionally evaluate against ground truth labels if available.

        Args:
            assessment_item (str): The assessment item/column to use as the estimation target.
            experiment_dir (Path): Path to the experiment directory containing trained models.
            with_labels (bool): Whether to evaluate against ground truth labels.
                               If True, evaluation metrics will be calculated and saved.
                               If False, only estimates will be generated.
        """

        model_checkpoint_dir = experiment_dir / assessment_item / "models"
        inference_log_dir = self.log_dir.with_name(f"inference_{self.log_dir.name}")
        result_base_dir = inference_log_dir / assessment_item / "inference_results"
        result_base_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        self.logger.info(
            f"[ Test | {assessment_item} ] Loading models from experiment directory: {experiment_dir.absolute()}"
        )
        with open(model_checkpoint_dir / "ord_rf_model.pkl", "rb") as f:
            ord_rf_model = pickle.load(f)
        with open(model_checkpoint_dir / "nom_rf_model.pkl", "rb") as f:
            nom_rf_model = pickle.load(f)

        data_handler = PatientDataHandler(
            self.config, self.logger, require_labels=with_labels
        )
        all_patients = data_handler.get_patient_data(
            assessment_item,
            self.config["ensemble"]["training"]["modality"][assessment_item],
        )

        self.logger.info(
            f"[ Test | {assessment_item} ] Total patients: {len(all_patients)}"
        )
        self.logger.info(
            f"[ Test | {assessment_item} ] Patients: {[p.name for p in all_patients]}"
        )

        test_patients = self.get_model_input(
            all_patients,
            self.config["ensemble"]["training"]["modality"][assessment_item],
            self.config["ensemble"]["training"]["pooling_dimension"][assessment_item],
        )

        # Test models
        pred_arrays = self.test(
            test_patients,
            ord_rf_model,
            nom_rf_model,
        )

        # Process results for each model
        model_metrics = {}
        model_names = ["ord_rf", "nom_rf", "ensemble"]

        for model_name in model_names:
            model_dir = result_base_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            if with_labels:
                # Calculate and save metrics
                model_metrics[model_name] = evaluate_metrics(
                    pred_arrays[model_name]["truth"], pred_arrays[model_name]["pred"]
                )

                metrics_dict = model_metrics[model_name]
                metrics_df = pd.DataFrame([metrics_dict])
                metrics_df.to_csv(model_dir / "metrics.csv", index=False)

                # Log metrics
                metrics_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics_dict.items()]
                )
                self.logger.info(
                    f"[ Test | {assessment_item} ] {model_name} - {metrics_str}"
                )

            else:
                pred_arrays[model_name]["truth"] = None

            # Create and save estimation DataFrame
            pred_df = self.create_prediction_df(
                pred_arrays[model_name]["patient_names"],
                pred_arrays[model_name]["pred"],
                pred_arrays[model_name]["truth"],
                assessment_item,
            )
            pred_df.to_csv(model_dir / "preds.csv", index=False)

        # Create summary comparison if labels are available
        if with_labels:
            # Create a summary file that compares all models
            summary_metrics = []
            for model_name in model_names:
                model_metrics_dict = model_metrics[model_name]
                model_metrics_dict["Model"] = model_name
                summary_metrics.append(model_metrics_dict)

                # Create and save summary DataFrame
                summary_df = pd.DataFrame(summary_metrics)
                if not summary_df.empty:
                    summary_df = summary_df.set_index("Model")
                    summary_df.to_csv(result_base_dir / "model_comparison_summary.csv")
