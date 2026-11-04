from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for model selection and repeated evaluation."""

    dataset_names: tuple[str, ...] = (
        "credit_risk_openml",
        "bank_marketing_openml",
    )
    n_runs: int = 21
    test_size: float = 0.20
    validation_size: float = 0.20
    random_seed: int = 42
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("data/cache")

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return deterministic seeds used in repeated evaluation."""

        return tuple(self.random_seed + offset for offset in range(self.n_runs))

    def validate(self) -> None:
        """Validate configuration values before execution."""

        if self.n_runs < 2:
            raise ValueError("n_runs must be at least 2.")
        if not 0.05 <= self.test_size <= 0.40:
            raise ValueError("test_size must be between 0.05 and 0.40.")
        if not 0.05 <= self.validation_size <= 0.40:
            raise ValueError("validation_size must be between 0.05 and 0.40.")
        if self.test_size + self.validation_size >= 0.60:
            raise ValueError("The combined validation and test fractions must be below 0.60.")
