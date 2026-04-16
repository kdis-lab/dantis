# Experimental TSAD Benchmark

This folder contains the minimal infrastructure to run reproducible anomaly-detection benchmarks on time-series datasets, starting with the UCR Anomaly Archive.

## Environment

Use the DANTIS environment or create a compatible one with the repository requirements:

```bash
conda activate dantis
pip install -r requirement.txt
```

## Run a benchmark

Default UCR benchmark configuration:

```bash
python -m experiment.run_benchmark \
  --benchmark-config experiment/config/benchmark_ucr.json \
  --algorithms-config experiment/config/algorithms.json
```

You can override the dataset root, output directory, seed, and selected algorithms:

```bash
python -m experiment.run_benchmark \
  --dataset-dir experiment/datasets/UCR_Anomaly \
  --output-dir experiment/results/ucr_benchmark \
  --seed 42 \
  --algorithms lof iforest hbos
```

## Configuration

- `experiment/config/algorithms.json` defines the algorithm registry, effective hyperparameters, and threshold policy.
- `experiment/config/benchmark_ucr.json` selects the collection, dataset folder, output folder, seed, and default algorithm subset.

## Output semantics

Algorithms do not share a single output contract. The benchmark runner uses registry metadata to enforce how each algorithm is handled:

- `output_mode="label"`: `predict()` must return binary labels and no extra thresholding is applied.
- `output_mode="score"`: scores are extracted from ordered methods (for example `decision_function`, `get_anomaly_score`, `score_samples`) and thresholding is applied.
- `output_mode="score_or_label"`: score methods are preferred first, with label fallback if needed.

The serialized `result.json` now includes traceability fields:

- `score_source`
- `threshold_mode`
- `threshold_value`
- `output_mode_used`
- `algorithm_status`
- `input_mode_used`

Thresholding modes are explicit and recorded per run:

- `contamination`
- `percentile`
- `fixed`

If an algorithm already returns binary labels, thresholding is skipped.

## Algorithm status

Initial registry status split:

- `validated`: `lof`, `ocsvm`, `iforest`, `hbos`, `ssa`, `arima`, `vae`
- `experimental`: `kmeans`, `dbstream`, `rgraph`, `eif`, `grammarviz`, `hotsax`, `mp_damp`, `deepant`, `telemanom`
- `disabled`: reserved for algorithms with confirmed dependency incompatibility in the active environment

## Output layout

- Per algorithm-dataset run: `result.json`, `y_pred.npy`, `y_score.npy` (score mode only)
- Aggregated summary: `benchmark_results.csv`
- Optional parquet summary: `benchmark_results.parquet`
- Friedman-ready matrix: `f1_matrix.csv`

## Notes

- UCR anomaly filenames are parsed as `dataset_id`, `dataset_name`, `train_end`, `anomaly_start`, `anomaly_end`.
- The loader treats `anomaly_end` as inclusive by default.
- Failed algorithm-dataset runs are recorded with explicit `status` and `error_message`, while the benchmark continues.
