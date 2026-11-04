from __future__ import annotations

import pandas as pd

from classification_benchmark.statistics import cochran_q_tests, pairwise_mcnemar_tests


def _prediction_frame() -> pd.DataFrame:
    correctness = {
        "a": [1, 1, 1, 1, 1, 1, 1, 1],
        "b": [1, 1, 1, 1, 0, 0, 0, 0],
        "c": [1, 0, 0, 0, 0, 0, 0, 0],
    }
    rows = []
    for model, values in correctness.items():
        for sample_index, correct in enumerate(values):
            rows.append(
                {
                    "dataset": "fixture",
                    "model": model,
                    "model_display_name": model.upper(),
                    "sample_index": sample_index,
                    "correct": correct,
                }
            )
    return pd.DataFrame(rows)


def test_cochran_q_returns_global_comparison() -> None:
    result = cochran_q_tests(_prediction_frame())
    assert len(result) == 1
    assert result.loc[0, "n_models"] == 3
    assert 0.0 <= result.loc[0, "p_value"] <= 1.0


def test_mcnemar_returns_holm_adjustment() -> None:
    result = pairwise_mcnemar_tests(_prediction_frame())
    assert len(result) == 3
    assert "p_value_holm" in result.columns
    assert "discordant_effect" in result.columns
