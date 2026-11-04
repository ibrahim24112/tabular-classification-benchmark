from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted.tolist()


def cochran_q_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare multiple classifiers on paired final-test correctness with Cochran's Q."""

    rows: list[dict[str, float | str | int | bool]] = []
    for dataset, dataset_frame in predictions.groupby("dataset"):
        pivot = dataset_frame.pivot(
            index="sample_index",
            columns="model",
            values="correct",
        ).dropna()
        n_samples, n_models = pivot.shape
        if n_models < 3 or n_samples == 0:
            continue

        matrix = pivot.to_numpy(dtype=float)
        column_totals = matrix.sum(axis=0)
        row_totals = matrix.sum(axis=1)
        total = float(column_totals.sum())
        denominator = n_models * total - float(np.sum(row_totals**2))
        if denominator <= 0:
            statistic = 0.0
            p_value = 1.0
        else:
            numerator = (n_models - 1) * (
                n_models * float(np.sum(column_totals**2)) - total**2
            )
            statistic = numerator / denominator
            p_value = float(chi2.sf(statistic, df=n_models - 1))

        rows.append(
            {
                "dataset": dataset,
                "n_samples": int(n_samples),
                "n_models": int(n_models),
                "statistic": float(statistic),
                "degrees_of_freedom": int(n_models - 1),
                "p_value": p_value,
                "significant_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def pairwise_mcnemar_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """Run exact paired McNemar tests with Holm correction on the final test set."""

    all_rows: list[dict[str, float | str | int | bool]] = []
    for dataset, dataset_frame in predictions.groupby("dataset"):
        correctness = dataset_frame.pivot(
            index="sample_index",
            columns="model",
            values="correct",
        ).dropna()
        display_names = (
            dataset_frame[["model", "model_display_name"]]
            .drop_duplicates()
            .set_index("model")["model_display_name"]
            .to_dict()
        )
        pair_rows: list[dict[str, float | str | int]] = []
        raw_p_values: list[float] = []

        for model_a, model_b in combinations(correctness.columns.tolist(), 2):
            values_a = correctness[model_a].to_numpy(dtype=int)
            values_b = correctness[model_b].to_numpy(dtype=int)
            a_only = int(np.sum((values_a == 1) & (values_b == 0)))
            b_only = int(np.sum((values_a == 0) & (values_b == 1)))
            discordant = a_only + b_only
            p_value = (
                float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
                if discordant > 0
                else 1.0
            )
            raw_p_values.append(p_value)
            pair_rows.append(
                {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_a_display_name": display_names[model_a],
                    "model_b": model_b,
                    "model_b_display_name": display_names[model_b],
                    "n_samples": int(len(correctness)),
                    "a_correct_b_wrong": a_only,
                    "a_wrong_b_correct": b_only,
                    "discordant_pairs": discordant,
                    "discordant_effect": (
                        float((a_only - b_only) / discordant) if discordant else 0.0
                    ),
                    "p_value": p_value,
                }
            )

        adjusted = _holm_adjust(raw_p_values)
        for row, adjusted_p in zip(pair_rows, adjusted, strict=False):
            row["p_value_holm"] = float(adjusted_p)
            row["significant_0_05"] = bool(adjusted_p < 0.05)
            all_rows.append(row)

    return pd.DataFrame(all_rows)
