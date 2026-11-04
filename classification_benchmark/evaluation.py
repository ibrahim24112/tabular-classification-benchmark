from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_benchmark.config import ExperimentConfig
from classification_benchmark.datasets import DatasetBundle
from classification_benchmark.metrics import binary_classification_metrics
from classification_benchmark.models import ModelSpec
from classification_benchmark.selection import SelectedConfiguration, build_model_pipeline


@dataclass
class ReferenceModel:
    """Fitted model and predictions from the isolated final test set."""

    dataset: DatasetBundle
    model_spec: ModelSpec
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    scores: pd.Series
    threshold: float

    @property
    def predictions(self) -> pd.Series:
        """Return binary predictions at the selected validation threshold."""

        return pd.Series(
            (self.scores.to_numpy() >= self.threshold).astype(int),
            index=self.scores.index,
            name="prediction",
        )


def fixed_development_test_split(
    dataset: DatasetBundle,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the fixed development and final test partitions."""

    return train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=dataset.y,
    )


def evaluate_repeated_runs(
    datasets: list[DatasetBundle],
    model_specs: tuple[ModelSpec, ...],
    selected: dict[tuple[str, str], SelectedConfiguration],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate fixed configurations on paired development splits for stability analysis."""

    result_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        X_development, _, y_development, _ = fixed_development_test_split(dataset, config)
        relative_validation_size = config.validation_size / (1.0 - config.test_size)

        for run, seed in enumerate(config.seeds, start=1):
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_development,
                y_development,
                test_size=relative_validation_size,
                random_state=seed,
                stratify=y_development,
            )
            for model_spec in model_specs:
                chosen = selected[(dataset.name, model_spec.name)]
                pipeline = build_model_pipeline(
                    X_train,
                    model_spec,
                    chosen.parameters,
                    seed=seed,
                )
                pipeline.fit(X_train, y_train)
                scores = pipeline.predict_proba(X_validation)[:, 1]
                metrics = binary_classification_metrics(
                    y_validation.to_numpy(),
                    scores,
                    chosen.threshold,
                )
                result_rows.append(
                    {
                        "dataset": dataset.name,
                        "dataset_display_name": dataset.display_name,
                        "model": model_spec.name,
                        "model_display_name": model_spec.display_name,
                        "run": run,
                        "seed": seed,
                        "parameters": chosen.parameters,
                        **metrics,
                    }
                )

    return pd.DataFrame(result_rows)


def fit_reference_models(
    datasets: list[DatasetBundle],
    model_specs: tuple[ModelSpec, ...],
    selected: dict[tuple[str, str], SelectedConfiguration],
    config: ExperimentConfig,
) -> dict[tuple[str, str], ReferenceModel]:
    """Fit each selected model on development data and evaluate the isolated final test set."""

    fitted: dict[tuple[str, str], ReferenceModel] = {}
    for dataset in datasets:
        X_development, X_test, y_development, y_test = fixed_development_test_split(dataset, config)
        for model_spec in model_specs:
            chosen = selected[(dataset.name, model_spec.name)]
            pipeline = build_model_pipeline(
                X_development,
                model_spec,
                chosen.parameters,
                seed=config.random_seed,
            )
            pipeline.fit(X_development, y_development)
            scores = pd.Series(
                pipeline.predict_proba(X_test)[:, 1],
                index=X_test.index,
                name="score",
            )
            fitted[(dataset.name, model_spec.name)] = ReferenceModel(
                dataset=dataset,
                model_spec=model_spec,
                pipeline=pipeline,
                X_test=X_test,
                y_test=y_test,
                scores=scores,
                threshold=chosen.threshold,
            )
    return fitted


def build_reference_prediction_frame(
    models: dict[tuple[str, str], ReferenceModel],
) -> pd.DataFrame:
    """Create a paired final-test prediction table for classifier comparison tests."""

    rows: list[dict[str, Any]] = []
    for reference in models.values():
        predictions = reference.predictions
        for sample_index in reference.y_test.index:
            target = int(reference.y_test.loc[sample_index])
            prediction = int(predictions.loc[sample_index])
            rows.append(
                {
                    "dataset": reference.dataset.name,
                    "dataset_display_name": reference.dataset.display_name,
                    "model": reference.model_spec.name,
                    "model_display_name": reference.model_spec.display_name,
                    "sample_index": int(sample_index),
                    "y_true": target,
                    "y_pred": prediction,
                    "correct": int(target == prediction),
                }
            )
    return pd.DataFrame(rows)


def final_test_metrics(
    models: dict[tuple[str, str], ReferenceModel],
) -> pd.DataFrame:
    """Calculate metrics for every model on the isolated final test set."""

    rows: list[dict[str, Any]] = []
    for reference in models.values():
        rows.append(
            {
                "dataset": reference.dataset.name,
                "dataset_display_name": reference.dataset.display_name,
                "model": reference.model_spec.name,
                "model_display_name": reference.model_spec.display_name,
                **binary_classification_metrics(
                    reference.y_test.to_numpy(),
                    reference.scores.to_numpy(),
                    reference.threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated validation metrics using mean and standard deviation."""

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
        "ks_statistic",
    ]
    summary = (
        results.groupby(
            ["dataset", "dataset_display_name", "model", "model_display_name"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary
