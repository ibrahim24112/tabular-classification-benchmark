from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the validation threshold that maximizes F1-score."""

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.50
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def specificity_score(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate the true negative rate."""

    tn, fp, _, _ = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Calculate the two-sample Kolmogorov-Smirnov statistic for model scores."""

    positive_scores = np.asarray(scores)[np.asarray(y_true) == 1]
    negative_scores = np.asarray(scores)[np.asarray(y_true) == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return float("nan"), float("nan")
    result = ks_2samp(
        positive_scores,
        negative_scores,
        alternative="two-sided",
        mode="auto",
    )
    return float(result.statistic), float(result.pvalue)


def ks_curve_data(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return empirical score CDFs and the maximum KS separation point."""

    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    positive_scores = np.sort(score_array[y_array == 1])
    negative_scores = np.sort(score_array[y_array == 0])
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("Both classes are required to build a KS curve.")

    thresholds = np.unique(np.concatenate(([0.0], score_array, [1.0])))
    positive_cdf = np.searchsorted(positive_scores, thresholds, side="right") / positive_scores.size
    negative_cdf = np.searchsorted(negative_scores, thresholds, side="right") / negative_scores.size
    distances = np.abs(negative_cdf - positive_cdf)
    index = int(np.argmax(distances))
    return (
        thresholds,
        positive_cdf,
        negative_cdf,
        float(distances[index]),
        float(thresholds[index]),
    )


def binary_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate threshold-based and probability-based binary metrics."""

    y_true_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    predictions = (score_array >= threshold).astype(int)
    ks_value, ks_pvalue = ks_statistic(y_true_array, score_array)
    return {
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, predictions)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "specificity": specificity_score(y_true_array, predictions),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, score_array)),
        "average_precision": float(average_precision_score(y_true_array, score_array)),
        "brier_score": float(brier_score_loss(y_true_array, score_array)),
        "ks_statistic": ks_value,
        "ks_pvalue": ks_pvalue,
        "threshold": float(threshold),
    }
