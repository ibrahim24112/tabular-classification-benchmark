from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


EstimatorFactory = Callable[[dict[str, Any], int], object]


@dataclass(frozen=True)
class ModelSpec:
    """Model definition and compact candidate parameter configurations."""

    name: str
    display_name: str
    scale_numeric: bool
    clip_numeric: bool
    candidate_parameters: tuple[dict[str, Any], ...]
    estimator_factory: EstimatorFactory

    def create_estimator(self, parameters: dict[str, Any], seed: int) -> object:
        """Instantiate the estimator with deterministic settings."""

        return self.estimator_factory(parameters, seed)


def _logistic_regression(parameters: dict[str, Any], seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(parameters["C"]),
        class_weight="balanced",
        max_iter=1200,
        solver="liblinear",
        random_state=seed,
    )


def _decision_tree(parameters: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=int(parameters["max_depth"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        criterion=str(parameters["criterion"]),
        class_weight="balanced",
        random_state=seed,
    )


def _random_forest(parameters: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=str(parameters["max_features"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _hist_gradient_boosting(
    parameters: dict[str, Any],
    seed: int,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(parameters["max_iter"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        l2_regularization=float(parameters["l2_regularization"]),
        class_weight="balanced",
        random_state=seed,
    )


def default_model_specs() -> tuple[ModelSpec, ...]:
    """Return the four classifiers used in the experiment."""

    return (
        ModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            scale_numeric=True,
            clip_numeric=True,
            candidate_parameters=(
                {"C": 0.10},
                {"C": 1.00},
                {"C": 10.00},
            ),
            estimator_factory=_logistic_regression,
        ),
        ModelSpec(
            name="decision_tree",
            display_name="Decision Tree",
            scale_numeric=False,
            clip_numeric=False,
            candidate_parameters=(
                {"max_depth": 3, "min_samples_leaf": 10, "criterion": "gini"},
                {"max_depth": 4, "min_samples_leaf": 15, "criterion": "entropy"},
                {"max_depth": 5, "min_samples_leaf": 20, "criterion": "gini"},
            ),
            estimator_factory=_decision_tree,
        ),
        ModelSpec(
            name="random_forest",
            display_name="Random Forest",
            scale_numeric=False,
            clip_numeric=False,
            candidate_parameters=(
                {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 4, "max_features": "sqrt"},
                {"n_estimators": 180, "max_depth": 12, "min_samples_leaf": 3, "max_features": "sqrt"},
                {"n_estimators": 220, "max_depth": None, "min_samples_leaf": 5, "max_features": "log2"},
            ),
            estimator_factory=_random_forest,
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            display_name="Histogram Gradient Boosting",
            scale_numeric=False,
            clip_numeric=False,
            candidate_parameters=(
                {"learning_rate": 0.05, "max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 0.10},
                {"learning_rate": 0.08, "max_iter": 140, "max_leaf_nodes": 23, "l2_regularization": 0.50},
                {"learning_rate": 0.10, "max_iter": 120, "max_leaf_nodes": 31, "l2_regularization": 1.00},
            ),
            estimator_factory=_hist_gradient_boosting,
        ),
    )
