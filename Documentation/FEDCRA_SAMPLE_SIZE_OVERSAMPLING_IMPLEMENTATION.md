# Sample Size, Oversampling, and Class Coverage Implementation

## Summary

This repository now supports controlled dataset creation for federated experiments where:

- `data_config.total_samples` is treated as the exact target dataset size.
- `imbalance.class_ratios` defines the target class distribution.
- `oversample` is used when the requested sample count per class exceeds available samples.
- All classes present in the raw dataset are preserved and used in training and evaluation.

## Implementation Location

- `src/fedLearn/fed_data.py`
  - `federated_data_dirichlet(...)`
  - `create_controlled_dataset(...)`

- `main_fed_config.py`
  - `evaluate_server_model(...)`
  - evaluation now computes per-class metrics explicitly for all known class labels.

## Behavior

1. `federated_data_dirichlet(...)`
   - Loads the raw dataset from `data_config.folder_name` and `data_config.file_name`.
   - If `class_ratios` is configured:
     - Calls `create_controlled_dataset(...)` with `total_samples`.
     - This returns a dataset of exactly `total_samples` samples.
   - If `class_ratios` is not configured and the dataset is larger than `total_samples`, it downsamples to `total_samples`.

2. `create_controlled_dataset(...)`
   - Validates the label column exists.
   - Normalizes label names before matching requested class ratios.
   - Drops any requested ratio labels that are not present in the dataset.
   - If no requested ratio labels match, it falls back to a uniform distribution over present classes.
   - Computes exact counts for each matched class and samples accordingly.
   - Uses sampling with replacement when a class has fewer raw examples than the requested count.
   - Ensures the returned dataset length equals `total_samples`.

## Sample Size and Oversampling Guarantees

- `total_samples` in `conf/config.yaml` is the target dataset size for federated split creation.
- When `imbalance.class_ratios` is provided, oversampling is enabled implicitly by the controlled dataset creator.
- This means the pipeline can generate a dataset of arbitrary configured size, even if the raw data would otherwise be too small for the requested class balance.
- All present classes are included in the created dataset, unless a requested class label is truly absent from the raw dataset.

## Class Coverage

- The code uses `LabelEncoder` after controlled dataset creation to derive `class_names`.
- This ensures all classes present in the dataset are assigned label IDs.
- `main_fed_config.py` now computes per-class precision, recall, and F1 explicitly for all expected label IDs.
- `server_metrics.json` therefore records metrics for every class in `class_names`, not just classes observed in predictions.

## Configuration Notes

- `conf/config.yaml` sets
  - `dataset.total_samples`
  - `data_config.total_samples`
  - `data_config.sample_size`
  - `imbalance.class_ratios`
- `conf/imbalance/*.yaml` defines the default class ratios for balanced, mild, severe, and extreme experiments.

## Practical Confirmation

Yes, the implementation supports:

- `total_samples` driven dataset sizing for experiments.
- oversampling when required to satisfy requested class ratios.
- full class coverage for classes present in the dataset.
- explicit per-class metric logging.

If you want this behavior for a different dataset shape, keep `imbalance.class_ratios` enabled and set `dataset.total_samples` to the experiment sample size you require.
