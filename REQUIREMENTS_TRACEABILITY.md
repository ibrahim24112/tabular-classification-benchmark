# Requirements traceability

| Assignment requirement | Project implementation |
|---|---|
| Define a relevant problem | Credit-risk classification and pre-contact bank-campaign targeting |
| Use real or simulated data | Two real UCI datasets; no simulated data |
| Describe data origin | UCI source, identifier, DOI, URL, records, predictors, and target documented |
| Analyze missing and categorical data | Notebook quality audit and data dictionaries |
| Provide exploratory analysis | Class balance, descriptive statistics, distributions, boxplots, correlations, and percentile extremes |
| Explain preprocessing | Leakage-safe pipelines, imputation, one-hot encoding, model-specific clipping/scaling |
| Evaluate at least three techniques | Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting |
| Evaluate parameter configurations | Three compact predefined configurations per model |
| Separate training, validation, and test | Fixed 60/20/20 conceptual split with isolated final test set |
| Repeat stochastic experiments | 21 paired development evaluations with deterministic seeds |
| Use suitable classification metrics | Balanced accuracy, precision, recall, specificity, F1, ROC-AUC, Average Precision, Brier score, and KS |
| Compare algorithms fairly | Same partitions, selected thresholds, and experimental conditions within each dataset |
| Present results with figures | Selected EDA, performance, ROC/PR, confusion/threshold, gains/lift/KS, importance, and tree figures |
| Use statistical comparison | Cochran Q and exact paired McNemar tests with Holm correction on final-test predictions |
| Provide organized executable code | Notebook, reusable modules, command-line script, README, and unit tests |
