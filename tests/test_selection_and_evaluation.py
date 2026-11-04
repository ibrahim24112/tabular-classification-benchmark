from __future__ import annotations

from dataclasses import replace

from classification_benchmark.config import ExperimentConfig
from classification_benchmark.evaluation import (
    build_reference_prediction_frame,
    evaluate_repeated_runs,
    fit_reference_models,
    summarize_results,
)
from classification_benchmark.models import default_model_specs
from classification_benchmark.selection import select_configurations


def test_compact_selection_and_repeated_evaluation(binary_dataset, tmp_path) -> None:
    config = ExperimentConfig(
        dataset_names=(binary_dataset.name,),
        n_runs=2,
        output_dir=tmp_path / "outputs",
    )
    compact_specs = tuple(
        replace(spec, candidate_parameters=(spec.candidate_parameters[0],))
        for spec in default_model_specs()[:3]
    )
    selection_frame, selected = select_configurations([binary_dataset], compact_specs, config)
    results = evaluate_repeated_runs(
        [binary_dataset],
        compact_specs,
        selected,
        config,
    )
    references = fit_reference_models([binary_dataset], compact_specs, selected, config)
    predictions = build_reference_prediction_frame(references)
    summary = summarize_results(results)

    assert len(selection_frame) == len(compact_specs)
    assert len(results) == config.n_runs * len(compact_specs)
    assert not predictions.empty
    assert not summary.empty
