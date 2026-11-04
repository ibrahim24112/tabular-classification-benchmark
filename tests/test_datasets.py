from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from classification_benchmark import datasets


def _openml_result(features: pd.DataFrame, target: pd.Series) -> SimpleNamespace:
    return SimpleNamespace(data=features, target=target)


def test_credit_loader_maps_bad_risk_to_positive(monkeypatch, tmp_path) -> None:
    rows = 10
    frame = pd.DataFrame(
        np.arange(rows * 20).reshape(rows, 20),
        columns=[f"Attribute{index}" for index in range(1, 21)],
    )
    target = pd.Series(["1"] * 7 + ["2"] * 3, name="class")
    monkeypatch.setattr(
        datasets,
        "fetch_openml",
        lambda **kwargs: _openml_result(frame, target),
    )

    bundle = datasets.load_credit_risk_openml(tmp_path)

    assert bundle.name == "credit_risk_openml"
    assert bundle.X.columns.tolist() == list(datasets.GERMAN_CREDIT_FEATURES)
    assert bundle.y.tolist() == [0] * 7 + [1] * 3
    assert bundle.positive_class == "bad"


def test_bank_loader_maps_yes_and_removes_duration(monkeypatch, tmp_path) -> None:
    rows = 20
    frame = pd.DataFrame(
        np.arange(rows * 16).reshape(rows, 16),
        columns=[f"V{index}" for index in range(1, 17)],
    )
    target = pd.Series(["no"] * 18 + ["yes"] * 2, name="y")
    monkeypatch.setattr(
        datasets,
        "fetch_openml",
        lambda **kwargs: _openml_result(frame, target),
    )

    bundle = datasets.load_bank_marketing_openml(tmp_path)

    assert bundle.name == "bank_marketing_openml"
    assert "duration" not in bundle.X.columns
    assert bundle.X.columns.tolist() == [
        name for name in datasets.BANK_MARKETING_FEATURES if name != "duration"
    ]
    assert bundle.y.tolist() == [0] * 18 + [1] * 2
    assert bundle.excluded_features == ("duration",)


def test_bank_numeric_labels_use_two_as_positive(monkeypatch, tmp_path) -> None:
    rows = 20
    frame = pd.DataFrame(
        np.arange(rows * 16).reshape(rows, 16),
        columns=[f"V{index}" for index in range(1, 17)],
    )
    target = pd.Series(["1"] * 18 + ["2"] * 2, name="y")
    monkeypatch.setattr(
        datasets,
        "fetch_openml",
        lambda **kwargs: _openml_result(frame, target),
    )

    bundle = datasets.load_bank_marketing_openml(tmp_path)

    assert bundle.y.mean() == 0.10


def test_registry_rejects_unknown_dataset() -> None:
    try:
        datasets.load_dataset("unknown")
    except KeyError as exc:
        assert "Available datasets" in str(exc)
    else:
        raise AssertionError("Expected KeyError for an unknown dataset.")
