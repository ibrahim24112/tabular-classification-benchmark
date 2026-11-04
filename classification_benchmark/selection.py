from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_benchmark.config import ExperimentConfig
from classification_benchmark.datasets import DatasetBundle
from classification_benchmark.metrics import binary_classification_metrics, select_f1_threshold
from classification_benchmark.models import ModelSpec
from classification_benchmark.preprocessing import build_preprocessor


@dataclass(frozen=True)
class SelectedConfiguration:
    """Selected model parameters and classification threshold."""

    dataset: str
    model: str
    parameters: dict[str, Any]
    threshold: float
    validation_f1: float
    validation_average_precision: float
    validation_roc_auc: float


def build_model_pipeline(
    frame: pd.DataFrame,
    model_spec: ModelSpec,
    parameters: dict[str, Any],
    seed: int,
) -> Pipeline:
    """Build a preprocessing and estimator pipeline."""

    preprocessor = build_preprocessor(
        frame,
        scale_numeric=model_spec.scale_numeric,
        clip_numeric=model_spec.clip_numeric,
    )
    estimator = model_spec.create_estimator(parameters, seed)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _selection_split(
    dataset: DatasetBundle,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_development, _, y_development, _ = train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=dataset.y,
    )
    relative_validation_size = config.validation_size / (1.0 - config.test_size)
    return train_test_split(
        X_development,
        y_development,
        test_size=relative_validation_size,
        random_state=config.random_seed,
        stratify=y_development,
    )


def select_configurations(
    datasets: list[DatasetBundle],
    model_specs: tuple[ModelSpec, ...],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[tuple[str, str], SelectedConfiguration]]:
    """Select compact model configurations on one fixed validation split."""

    config.validate()
    rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], SelectedConfiguration] = {}

    for dataset in datasets:
        X_train, X_validation, y_train, y_validation = _selection_split(dataset, config)
        for model_spec in model_specs:
            model_rows: list[dict[str, Any]] = []
            for candidate_index, parameters in enumerate(model_spec.candidate_parameters, start=1):
                pipeline = build_model_pipeline(
                    X_train,
                    model_spec,
                    parameters,
                    seed=config.random_seed,
                )
                pipeline.fit(X_train, y_train)
                scores = pipeline.predict_proba(X_validation)[:, 1]
                threshold = select_f1_threshold(y_validation.to_numpy(), scores)
                metrics = binary_classification_metrics(
                    y_validation.to_numpy(),
                    scores,
                    threshold,
                )
                row = {
                    "dataset": dataset.name,
                    "dataset_display_name": dataset.display_name,
                    "model": model_spec.name,
                    "model_display_name": model_spec.display_name,
                    "candidate": candidate_index,
                    "parameters": parameters,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
                rows.append(row)
                model_rows.append(row)

            best = max(
                model_rows,
                key=lambda item: (
                    item["validation_average_precision"],
                    item["validation_f1"],
                    item["validation_roc_auc"],
                ),
            )
            selected[(dataset.name, model_spec.name)] = SelectedConfiguration(
                dataset=dataset.name,
                model=model_spec.name,
                parameters=dict(best["parameters"]),
                threshold=float(best["validation_threshold"]),
                validation_f1=float(best["validation_f1"]),
                validation_average_precision=float(best["validation_average_precision"]),
                validation_roc_auc=float(best["validation_roc_auc"]),
            )

    return pd.DataFrame(rows), selected
