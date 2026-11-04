from __future__ import annotations

from pathlib import Path
import sys


def find_project_root(start: Path) -> Path:
    """Find the project root from the script location."""

    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "classification_benchmark").is_dir() and (candidate / "requirements.txt").exists():
            return candidate
    raise RuntimeError("Project root not found.")


PROJECT_ROOT = find_project_root(Path(__file__).parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification_benchmark.config import ExperimentConfig
from classification_benchmark.datasets import load_dataset
from classification_benchmark.evaluation import (
    build_reference_prediction_frame,
    evaluate_repeated_runs,
    final_test_metrics,
    fit_reference_models,
    summarize_results,
)
from classification_benchmark.io import ensure_figure_directory
from classification_benchmark.models import default_model_specs
from classification_benchmark.selection import select_configurations
from classification_benchmark.statistics import cochran_q_tests, pairwise_mcnemar_tests
from classification_benchmark.visualization import (
    plot_decision_tree_model,
    plot_discrimination_curves,
    plot_operating_point_diagnostics,
    plot_performance_overview,
    plot_permutation_importance,
    plot_ranking_diagnostics,
)


def main() -> None:
    """Run the numerical workflow and save only selected analytical figures."""

    config = ExperimentConfig(
        output_dir=PROJECT_ROOT / "outputs",
        cache_dir=PROJECT_ROOT / "data" / "cache",
    )
    config.validate()
    figure_dir = ensure_figure_directory(config.output_dir)

    datasets = [load_dataset(name, config.cache_dir) for name in config.dataset_names]
    model_specs = default_model_specs()

    selection_results, selected = select_configurations(datasets, model_specs, config)
    repeated_results = evaluate_repeated_runs(datasets, model_specs, selected, config)
    summary = summarize_results(repeated_results)
    reference_models = fit_reference_models(datasets, model_specs, selected, config)
    test_metrics = final_test_metrics(reference_models)
    prediction_frame = build_reference_prediction_frame(reference_models)

    print("\nSelected configurations")
    print(selection_results.sort_values(["dataset", "model", "candidate"]).to_string(index=False))
    print("\nRepeated-run summary")
    print(summary.to_string(index=False))
    print("\nFinal-test metrics")
    print(test_metrics.to_string(index=False))
    print("\nCochran Q")
    print(cochran_q_tests(prediction_frame).to_string(index=False))
    print("\nPairwise McNemar with Holm correction")
    print(pairwise_mcnemar_tests(prediction_frame).to_string(index=False))

    best_model_names = (
        summary.sort_values(["dataset", "f1_mean"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
        .set_index("dataset")["model"]
        .to_dict()
    )

    for dataset in datasets:
        dataset_models = {
            model_spec.name: reference_models[(dataset.name, model_spec.name)]
            for model_spec in model_specs
        }
        best_reference = reference_models[(dataset.name, best_model_names[dataset.name])]
        decision_tree_reference = reference_models[(dataset.name, "decision_tree")]

        plot_performance_overview(
            summary,
            repeated_results,
            dataset.name,
            figure_dir / f"{dataset.name}_performance_overview.png",
        )
        plot_discrimination_curves(
            dataset_models,
            figure_dir / f"{dataset.name}_discrimination_curves.png",
        )
        plot_operating_point_diagnostics(
            best_reference,
            figure_dir / f"{dataset.name}_operating_point.png",
        )
        plot_ranking_diagnostics(
            best_reference,
            figure_dir / f"{dataset.name}_ranking_diagnostics.png",
        )
        plot_permutation_importance(
            best_reference,
            figure_dir / f"{dataset.name}_permutation_importance.png",
        )
        plot_decision_tree_model(
            decision_tree_reference,
            figure_dir / f"{dataset.name}_decision_tree.png",
        )

    print(f"\nExecution completed. Selected figures are available at: {figure_dir}")


if __name__ == "__main__":
    main()
