from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


GERMAN_CREDIT_FEATURES = (
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
)

BANK_MARKETING_FEATURES = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
)


@dataclass(frozen=True)
class DatasetBundle:
    """Feature matrix, target vector, and metadata for a binary classification task."""

    name: str
    display_name: str
    description: str
    source: str
    source_url: str
    X: pd.DataFrame
    y: pd.Series
    target_names: tuple[str, str]
    positive_class: str
    excluded_features: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        """Return the number of observations."""

        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Return the number of predictors."""

        return int(self.X.shape[1])


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry for an available dataset."""

    name: str
    display_name: str
    loader: Callable[[Path], DatasetBundle]


def _safe_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "feature"


def _normalize_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [_safe_column_name(column) for column in result.columns]
    return result


def _rename_features_by_position(
    frame: pd.DataFrame,
    expected_names: tuple[str, ...],
) -> pd.DataFrame:
    if frame.shape[1] != len(expected_names):
        raise ValueError(
            f"Expected {len(expected_names)} predictors, but received {frame.shape[1]}."
        )
    result = frame.copy()
    result.columns = list(expected_names)
    return result


def _encode_explicit_binary_target(
    target: pd.Series,
    positive_values: set[str],
    negative_values: set[str],
    name: str,
) -> pd.Series:
    normalized = target.astype(str).str.strip().str.lower()
    positive = {str(value).strip().lower() for value in positive_values}
    negative = {str(value).strip().lower() for value in negative_values}
    observed = set(normalized.dropna().unique())
    unknown = observed.difference(positive | negative)
    if unknown:
        raise ValueError(f"Unexpected target values for {name}: {sorted(unknown)}")
    if not observed.intersection(positive) or not observed.intersection(negative):
        raise ValueError(f"Both target classes must be present for {name}.")
    encoded = normalized.isin(positive).astype(int)
    encoded.name = name
    return encoded


def _stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_samples:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=seed,
        stratify=y,
    )
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def _validate_positive_rate(y: pd.Series, lower: float, upper: float, dataset: str) -> None:
    rate = float(y.mean())
    if not lower <= rate <= upper:
        raise ValueError(
            f"Unexpected positive-class rate for {dataset}: {rate:.4f}. "
            "Review the target mapping before running the experiment."
        )


def load_credit_risk_openml(cache_dir: Path) -> DatasetBundle:
    """Load German Credit from OpenML through scikit-learn."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(
        data_id=31,
        as_frame=True,
        data_home=str(cache_dir),
        parser="auto",
    )
    X = _rename_features_by_position(
        _normalize_frame_columns(data.data),
        GERMAN_CREDIT_FEATURES,
    )
    y = _encode_explicit_binary_target(
        pd.Series(data.target, name="credit_risk"),
        positive_values={"bad", "2"},
        negative_values={"good", "1"},
        name="credit_risk",
    )
    _validate_positive_rate(y, lower=0.20, upper=0.40, dataset="German Credit Risk")
    return DatasetBundle(
        name="credit_risk_openml",
        display_name="German Credit Risk",
        description="Classifies applicants as good or bad credit risks using financial and demographic attributes.",
        source="OpenML data_id=31 loaded with sklearn.datasets.fetch_openml; original Statlog German Credit dataset",
        source_url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        X=X,
        y=y,
        target_names=("good", "bad"),
        positive_class="bad",
    )


def load_bank_marketing_openml(cache_dir: Path) -> DatasetBundle:
    """Load Bank Marketing from OpenML for a pre-contact targeting scenario."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(
        data_id=1461,
        as_frame=True,
        data_home=str(cache_dir),
        parser="auto",
    )
    X = _rename_features_by_position(
        _normalize_frame_columns(data.data),
        BANK_MARKETING_FEATURES,
    )
    y = _encode_explicit_binary_target(
        pd.Series(data.target, name="subscription"),
        positive_values={"yes", "2", "true"},
        negative_values={"no", "1", "false"},
        name="subscription",
    )
    _validate_positive_rate(y, lower=0.05, upper=0.25, dataset="Bank Marketing")

    excluded_features = ("duration",)
    X = X.drop(columns=list(excluded_features))
    X, y = _stratified_sample(X, y, max_samples=6000, seed=42)

    return DatasetBundle(
        name="bank_marketing_openml",
        display_name="Bank Marketing",
        description="Predicts term-deposit subscription before the campaign call outcome is known.",
        source="OpenML data_id=1461 loaded with sklearn.datasets.fetch_openml; stratified sample of up to 6,000 observations",
        source_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
        X=X,
        y=y,
        target_names=("no", "yes"),
        positive_class="yes",
        excluded_features=excluded_features,
    )


_DATASETS: dict[str, DatasetSpec] = {
    "credit_risk_openml": DatasetSpec(
        "credit_risk_openml",
        "German Credit Risk",
        load_credit_risk_openml,
    ),
    "bank_marketing_openml": DatasetSpec(
        "bank_marketing_openml",
        "Bank Marketing",
        load_bank_marketing_openml,
    ),
}


def list_dataset_specs() -> tuple[DatasetSpec, ...]:
    """Return all registered dataset specifications."""

    return tuple(_DATASETS.values())


def load_dataset(
    name: str,
    cache_dir: Path | str = Path("data/cache"),
) -> DatasetBundle:
    """Load a registered dataset by name."""

    try:
        spec = _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASETS))
        raise KeyError(f"Dataset '{name}' is not registered. Available datasets: {available}") from exc
    try:
        return spec.loader(Path(cache_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Dataset '{name}' could not be loaded from OpenML. "
            "Check the internet connection and review the original error."
        ) from exc
