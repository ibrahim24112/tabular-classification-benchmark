from __future__ import annotations

import numpy as np

from classification_benchmark.metrics import (
    binary_classification_metrics,
    ks_curve_data,
    ks_statistic,
    select_f1_threshold,
)


def test_threshold_is_within_probability_range() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = select_f1_threshold(target, scores)
    assert 0.0 <= threshold <= 1.0


def test_binary_metrics_include_core_measures() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = binary_classification_metrics(target, scores, threshold=0.5)
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["ks_statistic"] == 1.0


def test_ks_statistic_separates_score_distributions() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    statistic, p_value = ks_statistic(target, scores)
    assert statistic == 1.0
    assert 0.0 <= p_value <= 1.0


def test_ks_curve_returns_maximum_separation() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thresholds, positive_cdf, negative_cdf, statistic, threshold = ks_curve_data(target, scores)
    assert len(thresholds) == len(positive_cdf) == len(negative_cdf)
    assert statistic == 1.0
    assert 0.0 <= threshold <= 1.0
