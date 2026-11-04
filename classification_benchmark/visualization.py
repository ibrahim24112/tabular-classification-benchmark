from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.tree import plot_tree

from classification_benchmark.evaluation import ReferenceModel
from classification_benchmark.metrics import ks_curve_data


BLUE = "#4C78A8"
LIGHT_BLUE = "#A9CBE3"
DEEP_BLUE = "#2F5D7C"
GREEN = "#59A14F"
LIGHT_GREEN = "#B9DDB5"
DEEP_GREEN = "#3B7D43"
TEAL = "#4E9F9A"
PALETTE = (BLUE, GREEN, TEAL, LIGHT_BLUE)


def _prepare_path(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _save_show_close(fig: plt.Figure, path: Path | str) -> Path:
    destination = _prepare_path(path)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend or "inline" in backend:
        plt.show()
    plt.close(fig)
    return destination


def _blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color))
    return tuple(1.0 - strength * (1.0 - base))


def plot_performance_overview(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    dataset: str,
    path: Path | str,
) -> Path:
    """Plot mean F1 with uncertainty and repeated-run F1 distributions."""

    summary_frame = summary[summary["dataset"] == dataset].sort_values("f1_mean")
    result_frame = results[results["dataset"] == dataset]
    models = summary_frame["model_display_name"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(summary_frame))]
    axes[0].barh(
        summary_frame["model_display_name"],
        summary_frame["f1_mean"],
        xerr=summary_frame["f1_std"],
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    axes[0].set_xlabel("F1-score")
    axes[0].set_title("Mean F1 across 21 paired runs")
    axes[0].grid(axis="x", alpha=0.20)

    values = [
        result_frame.loc[result_frame["model_display_name"] == model, "f1"].to_numpy()
        for model in models
    ]
    box = axes[1].boxplot(
        values,
        tick_labels=models,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": DEEP_BLUE, "linewidth": 1.8},
        meanprops={
            "marker": "D",
            "markerfacecolor": DEEP_GREEN,
            "markeredgecolor": DEEP_GREEN,
            "markersize": 4,
        },
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[index % len(PALETTE)])
        patch.set_alpha(0.70)
    for element in ("whiskers", "caps"):
        for artist in box[element]:
            artist.set_color(DEEP_BLUE)
    axes[1].set_ylabel("F1-score")
    axes[1].set_title("F1 stability across repeated runs")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.20)
    return _save_show_close(fig, path)


def plot_discrimination_curves(
    models: Mapping[str, ReferenceModel],
    path: Path | str,
) -> Path:
    """Plot ROC and precision-recall curves for the isolated final test set."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for index, reference in enumerate(models.values()):
        color = PALETTE[index % len(PALETTE)]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            reference.y_test,
            reference.scores,
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{reference.model_spec.display_name} (AUC={roc_auc:.3f})",
            color=color,
            linewidth=2,
        )

        precision, recall, _ = precision_recall_curve(
            reference.y_test,
            reference.scores,
        )
        pr_auc = auc(recall, precision)
        axes[1].plot(
            recall,
            precision,
            label=f"{reference.model_spec.display_name} (AUC={pr_auc:.3f})",
            color=color,
            linewidth=2,
        )

    axes[0].plot([0, 1], [0, 1], linestyle="--", color=LIGHT_BLUE, linewidth=1.2)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.20)

    prevalence = float(next(iter(models.values())).y_test.mean())
    axes[1].axhline(
        prevalence,
        linestyle="--",
        color=LIGHT_GREEN,
        linewidth=1.2,
        label=f"Positive rate={prevalence:.3f}",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curves")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.20)
    return _save_show_close(fig, path)


def _threshold_metrics(reference: ReferenceModel) -> pd.DataFrame:
    thresholds = np.linspace(0.02, 0.98, 97)
    y_true = reference.y_test.to_numpy()
    scores = reference.scores.to_numpy()
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        true_positive = np.sum((y_true == 1) & (predictions == 1))
        true_negative = np.sum((y_true == 0) & (predictions == 0))
        false_positive = np.sum((y_true == 0) & (predictions == 1))
        false_negative = np.sum((y_true == 1) & (predictions == 0))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def plot_operating_point_diagnostics(
    reference: ReferenceModel,
    path: Path | str,
) -> Path:
    """Plot normalized confusion matrix and threshold trade-offs for one model."""

    predictions = reference.predictions.to_numpy()
    matrix = confusion_matrix(
        reference.y_test,
        predictions,
        labels=[0, 1],
        normalize="true",
    )
    tradeoff = _threshold_metrics(reference)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    image = axes[0].imshow(matrix, cmap="GnBu", vmin=0.0, vmax=1.0)
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.1%}",
                ha="center",
                va="center",
                color=DEEP_BLUE,
                fontsize=13,
                fontweight="bold",
            )
    axes[0].set_xticks([0, 1], labels=reference.dataset.target_names)
    axes[0].set_yticks([0, 1], labels=reference.dataset.target_names)
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("Actual class")
    axes[0].set_title("Normalized confusion matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(tradeoff["threshold"], tradeoff["precision"], label="Precision", color=BLUE, linewidth=2)
    axes[1].plot(tradeoff["threshold"], tradeoff["recall"], label="Recall", color=GREEN, linewidth=2)
    axes[1].plot(tradeoff["threshold"], tradeoff["specificity"], label="Specificity", color=LIGHT_BLUE, linewidth=2)
    axes[1].plot(tradeoff["threshold"], tradeoff["f1"], label="F1-score", color=TEAL, linewidth=2)
    axes[1].axvline(
        reference.threshold,
        linestyle="--",
        color=DEEP_BLUE,
        label=f"Selected threshold={reference.threshold:.2f}",
    )
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Operating-point trade-off")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.20)

    fig.suptitle(
        f"Final-test diagnostics - {reference.model_spec.display_name}",
        y=1.02,
        fontsize=14,
    )
    return _save_show_close(fig, path)


def _ordered_gain_data(reference: ReferenceModel) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "target": reference.y_test.to_numpy(),
            "score": reference.scores.to_numpy(),
        }
    ).sort_values("score", ascending=False, ignore_index=True)
    population = np.arange(1, len(frame) + 1) / len(frame)
    positive_total = max(int(frame["target"].sum()), 1)
    gains = frame["target"].cumsum().to_numpy() / positive_total
    return population, gains


def plot_ranking_diagnostics(
    reference: ReferenceModel,
    path: Path | str,
) -> Path:
    """Plot cumulative gains, lift, and the empirical KS score distributions."""

    population, gains = _ordered_gain_data(reference)
    lift = np.divide(gains, population, out=np.ones_like(gains), where=population > 0)
    thresholds, positive_cdf, negative_cdf, ks_value, ks_threshold = ks_curve_data(
        reference.y_test.to_numpy(),
        reference.scores.to_numpy(),
    )
    threshold_index = int(np.argmin(np.abs(thresholds - ks_threshold)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(population, gains, color=BLUE, linewidth=2, label=reference.model_spec.display_name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=LIGHT_GREEN, label="Random selection")
    axes[0].set_xlabel("Population proportion")
    axes[0].set_ylabel("Positive cases captured")
    axes[0].set_title("Cumulative gains")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.20)

    axes[1].plot(population, lift, color=GREEN, linewidth=2)
    axes[1].axhline(1.0, linestyle="--", color=LIGHT_BLUE, label="Random selection")
    axes[1].set_xlabel("Population proportion")
    axes[1].set_ylabel("Lift")
    axes[1].set_title("Lift curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.20)

    axes[2].plot(thresholds, negative_cdf, color=BLUE, linewidth=2, label=reference.dataset.target_names[0])
    axes[2].plot(thresholds, positive_cdf, color=GREEN, linewidth=2, label=reference.dataset.target_names[1])
    axes[2].vlines(
        ks_threshold,
        positive_cdf[threshold_index],
        negative_cdf[threshold_index],
        color=TEAL,
        linestyle="--",
        linewidth=2,
        label=f"KS={ks_value:.3f} at {ks_threshold:.3f}",
    )
    axes[2].set_xlabel("Predicted probability threshold")
    axes[2].set_ylabel("Empirical cumulative proportion")
    axes[2].set_title("Kolmogorov-Smirnov curve")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.20)

    fig.suptitle(
        f"Ranking diagnostics - {reference.model_spec.display_name}",
        y=1.03,
        fontsize=14,
    )
    return _save_show_close(fig, path)


def plot_decision_tree_model(
    reference: ReferenceModel,
    path: Path | str,
    display_depth: int = 3,
) -> Path:
    """Plot the first levels of a fitted decision tree using only blue and green shades."""

    preprocessor = reference.pipeline.named_steps["preprocessor"]
    estimator = reference.pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 10))
    annotations = plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=list(reference.dataset.target_names),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=7,
        max_depth=display_depth,
        ax=ax,
    )

    negative_name, positive_name = reference.dataset.target_names
    for annotation in annotations:
        text = annotation.get_text().lower()
        predicts_positive = f"class = {positive_name}".lower() in text
        base_color = GREEN if predicts_positive else BLUE
        edge_color = DEEP_GREEN if predicts_positive else DEEP_BLUE
        patch = annotation.get_bbox_patch()
        if patch is None:
            continue
        patch.set_facecolor(_blend_with_white(base_color, 0.68))
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.0)

    ax.set_title(
        f"Decision Tree interpretation - {reference.dataset.display_name}\nFirst {display_depth} levels",
        fontsize=14,
    )
    return _save_show_close(fig, path)


def plot_permutation_importance(
    reference: ReferenceModel,
    path: Path | str,
    seed: int = 42,
    top_n: int = 12,
) -> tuple[Path, pd.DataFrame]:
    """Calculate and plot permutation importance on the isolated final test set."""

    result = permutation_importance(
        reference.pipeline,
        reference.X_test,
        reference.y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=seed,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": reference.X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    plot_frame = frame.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        plot_frame["feature"],
        plot_frame["importance_mean"],
        xerr=plot_frame["importance_std"],
        color=TEAL,
        alpha=0.82,
        capsize=3,
    )
    ax.axvline(0.0, color=LIGHT_BLUE, linewidth=1)
    ax.set_xlabel("Decrease in average precision after permutation")
    ax.set_ylabel("Feature")
    ax.set_title(f"Permutation importance - {reference.model_spec.display_name}")
    ax.grid(axis="x", alpha=0.20)
    return _save_show_close(fig, path), frame
