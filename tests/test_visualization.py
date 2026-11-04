from __future__ import annotations

from dataclasses import replace

from classification_benchmark.config import ExperimentConfig
from classification_benchmark.evaluation import fit_reference_models
from classification_benchmark.models import default_model_specs
from classification_benchmark.selection import select_configurations
from classification_benchmark.visualization import (
    plot_decision_tree_model,
    plot_operating_point_diagnostics,
    plot_ranking_diagnostics,
)


def test_selected_plots_are_created(binary_dataset, tmp_path) -> None:
    config = ExperimentConfig(dataset_names=(binary_dataset.name,), n_runs=2)
    tree_spec = replace(
        default_model_specs()[1],
        candidate_parameters=(default_model_specs()[1].candidate_parameters[0],),
    )
    _, selected = select_configurations([binary_dataset], (tree_spec,), config)
    references = fit_reference_models([binary_dataset], (tree_spec,), selected, config)
    reference = references[(binary_dataset.name, tree_spec.name)]

    tree_path = plot_decision_tree_model(reference, tmp_path / "tree.png")
    operating_path = plot_operating_point_diagnostics(reference, tmp_path / "operating.png")
    ranking_path = plot_ranking_diagnostics(reference, tmp_path / "ranking.png")

    assert tree_path.exists()
    assert operating_path.exists()
    assert ranking_path.exists()
