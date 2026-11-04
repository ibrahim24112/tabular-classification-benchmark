from __future__ import annotations

import numpy as np

from classification_benchmark.preprocessing import PercentileClipper, build_preprocessor


def test_percentile_clipper_limits_extreme_values() -> None:
    values = np.array([[0.0], [1.0], [2.0], [100.0]])
    clipper = PercentileClipper(0.0, 0.75).fit(values)
    transformed = clipper.transform(np.array([[200.0]]))
    assert transformed[0, 0] <= clipper.upper_bounds_[0]


def test_preprocessor_handles_mixed_features(binary_dataset) -> None:
    transformer = build_preprocessor(
        binary_dataset.X,
        scale_numeric=True,
        clip_numeric=True,
    )
    transformed = transformer.fit_transform(binary_dataset.X, binary_dataset.y)
    assert transformed.shape[0] == binary_dataset.n_samples
    assert np.isfinite(transformed).all()
