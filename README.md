# Tabular Classification Benchmark

A reproducible study comparing four classifiers on two real-world tabular datasets: Statlog German Credit and Bank Marketing. The project emphasizes careful preprocessing, validation-based model selection, held-out testing, diagnostic metrics, and statistical comparison.

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- Histogram Gradient Boosting

## Methodology

The workflow documents the datasets, audits records and features, performs exploratory analysis, and uses leakage-safe scikit-learn pipelines. Imputation and preprocessing are learned from training data; model selection is performed on validation data; the final test set is kept separate until the end. Evaluation includes balanced accuracy, precision, recall, specificity, F1, ROC-AUC, Average Precision, Brier score, and KS, plus threshold and feature-importance analyses. Statistical comparisons use paired predictions and multiple-comparison correction where appropriate.

## Repository layout

```text
classification_benchmark/       reusable Python modules
notebooks/classification_experiment.ipynb
scripts/run_experiment.py       command-line workflow
tests/                          unit tests
outputs/figures/                selected result figures
REFERENCES_AND_DATASETS.md      dataset and method citations
REQUIREMENTS_TRACEABILITY.md    requirements mapping
requirements.txt
```

The notebook outputs were cleared in this portfolio copy; running the notebook regenerates its displayed analysis. OpenML data is downloaded when needed and cached locally. The cache is deliberately not committed.

## Setup and use

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python scripts/run_experiment.py
```

For the notebook, start Jupyter from the repository root and open `notebooks/classification_experiment.ipynb`. Run the tests with `pytest -q`.

## Data and responsible use

Dataset citations and source links are recorded in `REFERENCES_AND_DATASETS.md`. Check the original dataset terms before redistributing data or publishing the repository. This project is an educational benchmarking exercise, not a production decision system.

## Publication note

The code and test suite were curated from the supplied archive but were not executed as part of this sorting task. Confirm dependencies and run the tests in a clean environment before public release.