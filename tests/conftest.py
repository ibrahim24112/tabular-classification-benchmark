from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification_benchmark.datasets import DatasetBundle


@pytest.fixture
def binary_dataset() -> DatasetBundle:
    """Create a small mixed-type fixture for unit tests."""

    rng = np.random.default_rng(42)
    size = 240
    numeric_a = rng.normal(size=size)
    numeric_b = rng.normal(size=size)
    category = rng.choice(["a", "b", "c"], size=size)
    logits = 1.2 * numeric_a - 0.8 * numeric_b + (category == "c") * 0.7
    probability = 1.0 / (1.0 + np.exp(-logits))
    target = (probability > np.quantile(probability, 0.65)).astype(int)
    frame = pd.DataFrame(
        {
            "numeric_a": numeric_a,
            "numeric_b": numeric_b,
            "category": category,
        }
    )
    frame.loc[0, "numeric_a"] = np.nan
    return DatasetBundle(
        name="fixture",
        display_name="Fixture",
        description="Unit-test fixture.",
        source="Local fixture",
        source_url="",
        X=frame,
        y=pd.Series(target, name="target"),
        target_names=("negative", "positive"),
        positive_class="positive",
    )
