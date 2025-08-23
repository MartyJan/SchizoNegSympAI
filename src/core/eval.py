from typing import Dict

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    root_mean_squared_error,
)


def evaluate_metrics(trueY: np.ndarray, predY: np.ndarray) -> Dict[str, float]:
    """
    Evaluate metrics for model predictions.

    Args:
        trueY (np.ndarray): Ground truth labels.
        predY (np.ndarray): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary of evaluated metrics.
    """
    metrics = {}
    metrics["ME"] = np.mean(predY - trueY)
    metrics["Weighted Kappa"] = cohen_kappa_score(trueY, predY, weights="quadratic")
    metrics["RMSE"] = root_mean_squared_error(trueY, predY)
    metrics["MAE"] = mean_absolute_error(trueY, predY)

    # Calculate PCC only if sample size >= 2
    if len(trueY) >= 2:
        metrics["PCC"] = stats.pearsonr(trueY, predY).statistic
    else:
        metrics["PCC"] = np.nan

    # Calculate ICC only if sample size >= 5
    if len(trueY) >= 5:
        patient_ids = np.arange(len(trueY))

        pred_df = pd.DataFrame(
            {"Patient": patient_ids, "Rater": ["Machine"] * len(trueY), "Score": predY}
        )

        label_df = pd.DataFrame(
            {"Patient": patient_ids, "Rater": ["Doctor"] * len(trueY), "Score": trueY}
        )

        combined_df = pd.concat([pred_df, label_df])

        icc_results = pg.intraclass_corr(
            data=combined_df, targets="Patient", raters="Rater", ratings="Score"
        )
        metrics["ICC"] = icc_results.loc[2, "ICC"]
    else:
        metrics["ICC"] = np.nan

    return metrics
