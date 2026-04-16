from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .registry import ALGORITHM_REGISTRY, AlgorithmSpec, create_algorithm, get_algorithm_spec, list_algorithms
from .results import BenchmarkResult, results_to_dataframe, save_aggregate_results, save_numpy_array, save_result_json
from .stats import export_metric_matrix
from .ucr_loader import load_ucr_anomaly_collection


logger = logging.getLogger(__name__)


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_metric(metric_fn, y_true, y_pred, *, default=None):
    try:
        return float(metric_fn(y_true, y_pred))
    except Exception:
        return default


def _safe_roc_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _safe_average_precision(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def _align_scores_direction(scores: np.ndarray, score_direction: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    direction = str(score_direction or "higher").lower()
    if direction in {"lower", "lower_is_more_anomalous", "inverse"}:
        return -scores
    return scores


def calibrate_threshold(scores: np.ndarray, threshold_config: dict[str, Any] | None, hyperparameters: dict[str, Any] | None = None) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    config = dict(threshold_config or {})
    hyperparameters = hyperparameters or {}
    mode = str(config.get("mode", "contamination")).lower()

    if mode == "fixed":
        if "value" not in config:
            raise ValueError("Fixed threshold mode requires threshold.value.")
        return float(config["value"])

    if mode == "percentile":
        percentile = float(config.get("percentile", 95.0))
        return float(np.percentile(scores, percentile))

    contamination = config.get("contamination", hyperparameters.get("contamination", 0.1))
    contamination = float(contamination)
    contamination = min(0.5, max(0.001, contamination))
    return float(np.percentile(scores, 100.0 * (1.0 - contamination)))


def _is_binary(values: np.ndarray) -> bool:
    unique_values = np.unique(values)
    return bool(np.all(np.isin(unique_values, [0, 1])))


def _extract_binary_labels(model, X: np.ndarray) -> tuple[np.ndarray, str]:
    if not hasattr(model, "predict"):
        raise AttributeError(f"Model {type(model).__name__} does not expose predict() for label output.")
    labels = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if not _is_binary(labels):
        raise ValueError("predict() did not return binary labels for a label-mode algorithm.")
    return labels.astype(int), "predict_binary"


def _extract_scores_by_preference(model, X: np.ndarray, spec: AlgorithmSpec) -> tuple[np.ndarray, str]:
    methods = list(spec.score_method_preference)
    if spec.allow_predict_as_score:
        methods.append("predict")

    for method_name in methods:
        if not hasattr(model, method_name):
            continue
        values = np.asarray(getattr(model, method_name)(X), dtype=float).reshape(-1)
        if method_name == "predict" and _is_binary(values):
            continue
        return values, method_name

    raise AttributeError(
        f"Model {type(model).__name__} does not provide any preferred score method: {methods}."
    )


def _resolve_output(
    *,
    model,
    X: np.ndarray,
    spec: AlgorithmSpec,
) -> tuple[str, np.ndarray | None, np.ndarray | None, str]:
    output_mode = str(spec.output_mode).lower()

    if output_mode == "label":
        labels, source = _extract_binary_labels(model, X)
        return "label", None, labels, source

    if output_mode == "score":
        scores, source = _extract_scores_by_preference(model, X, spec)
        return "score", scores, None, source

    if output_mode == "score_or_label":
        try:
            scores, source = _extract_scores_by_preference(model, X, spec)
            return "score", scores, None, source
        except Exception:
            labels, source = _extract_binary_labels(model, X)
            return "label", None, labels, source

    raise ValueError(f"Unsupported output_mode {spec.output_mode!r} for algorithm {spec.name!r}.")


def _fit_model(model, X_train: np.ndarray, y_train: np.ndarray | None):
    fit = getattr(model, "fit")
    if y_train is None:
        return fit(X_train)
    try:
        return fit(X_train, y_train)
    except TypeError:
        return fit(X_train)


def _resolve_algorithm_config(algorithm_name: str, benchmark_config: dict[str, Any], algorithms_config: dict[str, Any]) -> dict[str, Any]:
    selected = benchmark_config.get("algorithms", [])
    if algorithm_name not in selected:
        return {}
    entry = dict(algorithms_config.get(algorithm_name, {}))
    benchmark_threshold = benchmark_config.get("threshold")
    if benchmark_threshold and "threshold" not in entry:
        entry["threshold"] = benchmark_threshold
    return entry


def _threshold_mode_name(threshold_config: dict[str, Any] | None) -> str:
    if not threshold_config:
        return "contamination"
    return str(threshold_config.get("mode", "contamination")).lower()


def run_benchmark(
    benchmark_config_path: str | Path,
    *,
    algorithms_config_path: str | Path,
    dataset_dir: str | Path | None = None,
    selected_algorithms: list[str] | None = None,
    output_dir: str | Path | None = None,
    random_seed: int | None = None,
    anomaly_end_inclusive: bool = True,
) -> list[BenchmarkResult]:
    benchmark_config = _load_json(benchmark_config_path)
    algorithms_config = _load_json(algorithms_config_path)
    stats_config = dict(benchmark_config.get("stats", {}))

    collection_name = benchmark_config.get("collection_name", "UCR_Anomaly")
    dataset_dir = Path(dataset_dir or benchmark_config.get("dataset_dir", "experiment/datasets/UCR_Anomaly"))
    output_dir = Path(output_dir or benchmark_config.get("output_dir", "experiment/results"))
    selected_algorithms = selected_algorithms or benchmark_config.get("algorithms", list_algorithms())
    random_seed = int(random_seed if random_seed is not None else benchmark_config.get("random_seed", 42))

    set_global_seed(random_seed)

    datasets = load_ucr_anomaly_collection(dataset_dir, anomaly_end_inclusive=anomaly_end_inclusive)
    results: list[BenchmarkResult] = []

    for dataset in datasets:
        benchmark_dataset = dataset.to_benchmark_dataset()
        for algorithm_name in selected_algorithms:
            if algorithm_name not in ALGORITHM_REGISTRY:
                result = BenchmarkResult(
                    dataset=dataset.dataset_name,
                    collection=collection_name,
                    algorithm=algorithm_name,
                    hyperparameters_effective={},
                    fit_time=0.0,
                    predict_time=0.0,
                    precision=None,
                    recall=None,
                    f1=None,
                    average_precision=None,
                    roc_auc=None,
                    n_train=benchmark_dataset.n_train,
                    n_test=benchmark_dataset.n_test,
                    n_anomalies_test=benchmark_dataset.n_anomalies_test,
                    status="failed",
                    algorithm_status="unknown",
                    input_mode_used="unknown",
                    error_message=f"Unknown algorithm: {algorithm_name}",
                    metadata={
                        "dataset_id": dataset.dataset_id,
                        "source_file": dataset.source_file,
                    },
                )
                results.append(result)
                continue

            algorithm_config = _resolve_algorithm_config(algorithm_name, benchmark_config, algorithms_config)
            hyperparameters = dict(algorithm_config.get("hyperparameters", {}))
            threshold_config = algorithm_config.get("threshold")
            spec = get_algorithm_spec(algorithm_name)
            score_direction = algorithm_config.get("score_direction", spec.score_direction)

            run_dir = output_dir / collection_name / dataset.dataset_name / algorithm_name
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                if spec.status == "disabled":
                    raise RuntimeError("Algorithm is marked as disabled in the registry.")

                model = create_algorithm(algorithm_name, hyperparameters=hyperparameters)

                start_fit = time.perf_counter()
                _fit_model(model, benchmark_dataset.X_train, benchmark_dataset.y_train)
                fit_time = time.perf_counter() - start_fit

                start_predict = time.perf_counter()

                output_mode_used = spec.output_mode
                score_source: str | None = None
                threshold_value: float | None = None
                threshold_mode: str | None = None
                y_score_path: str | None = None

                resolved_mode, test_scores_raw, test_labels_raw, score_source = _resolve_output(
                    model=model,
                    X=benchmark_dataset.X_test,
                    spec=spec,
                )

                if resolved_mode == "label":
                    y_pred = np.asarray(test_labels_raw, dtype=int).reshape(-1)
                    test_scores = None
                    output_mode_used = "label"
                else:
                    train_mode, train_scores_raw, _, _ = _resolve_output(
                        model=model,
                        X=benchmark_dataset.X_train,
                        spec=spec,
                    )
                    if train_mode != "score" or train_scores_raw is None or test_scores_raw is None:
                        raise RuntimeError("Score-mode algorithm did not provide continuous scores consistently.")

                    train_scores = _align_scores_direction(train_scores_raw, score_direction)
                    test_scores = _align_scores_direction(test_scores_raw, score_direction)
                    output_mode_used = "score"

                    if spec.requires_threshold:
                        threshold_mode = _threshold_mode_name(threshold_config)
                        threshold_value = calibrate_threshold(train_scores, threshold_config, hyperparameters=hyperparameters)
                        y_pred = (test_scores >= threshold_value).astype(int)
                    elif spec.predict_returns_labels:
                        y_pred, score_source = _extract_binary_labels(model, benchmark_dataset.X_test)
                    else:
                        raise RuntimeError("Algorithm returns scores but is configured without threshold and without label predict().")

                predict_time = time.perf_counter() - start_predict

                precision = _safe_metric(lambda y_true, y_hat: precision_score(y_true, y_hat, zero_division=0), benchmark_dataset.y_test, y_pred, default=None)
                recall = _safe_metric(lambda y_true, y_hat: recall_score(y_true, y_hat, zero_division=0), benchmark_dataset.y_test, y_pred, default=None)
                f1 = _safe_metric(lambda y_true, y_hat: f1_score(y_true, y_hat, zero_division=0), benchmark_dataset.y_test, y_pred, default=None)
                if test_scores is None:
                    average_precision = None
                    roc_auc = None
                else:
                    average_precision = _safe_average_precision(benchmark_dataset.y_test, test_scores)
                    roc_auc = _safe_roc_auc(benchmark_dataset.y_test, test_scores)

                y_pred_path = save_numpy_array(run_dir / "y_pred.npy", y_pred)
                if test_scores is not None:
                    y_score_path = save_numpy_array(run_dir / "y_score.npy", test_scores)

                result = BenchmarkResult(
                    dataset=dataset.dataset_name,
                    collection=collection_name,
                    algorithm=algorithm_name,
                    hyperparameters_effective={
                        **hyperparameters,
                        "threshold": threshold_config or benchmark_config.get("threshold"),
                        "score_direction": score_direction,
                    },
                    fit_time=fit_time,
                    predict_time=predict_time,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    average_precision=average_precision,
                    roc_auc=roc_auc,
                    n_train=benchmark_dataset.n_train,
                    n_test=benchmark_dataset.n_test,
                    n_anomalies_test=benchmark_dataset.n_anomalies_test,
                    status="completed",
                    score_source=score_source,
                    threshold_mode=threshold_mode,
                    threshold_value=threshold_value,
                    output_mode_used=output_mode_used,
                    algorithm_status=spec.status,
                    input_mode_used=spec.input_mode,
                    threshold=threshold_value,
                    y_pred_path=y_pred_path,
                    y_score_path=y_score_path,
                    metadata=dict(benchmark_dataset.metadata),
                )
            except Exception as exc:
                result = BenchmarkResult(
                    dataset=dataset.dataset_name,
                    collection=collection_name,
                    algorithm=algorithm_name,
                    hyperparameters_effective=hyperparameters,
                    fit_time=0.0,
                    predict_time=0.0,
                    precision=None,
                    recall=None,
                    f1=None,
                    average_precision=None,
                    roc_auc=None,
                    n_train=benchmark_dataset.n_train,
                    n_test=benchmark_dataset.n_test,
                    n_anomalies_test=benchmark_dataset.n_anomalies_test,
                    status="failed",
                    algorithm_status=get_algorithm_spec(algorithm_name).status,
                    input_mode_used=get_algorithm_spec(algorithm_name).input_mode,
                    error_message=str(exc),
                    metadata=dict(benchmark_dataset.metadata),
                )

            save_result_json(run_dir / "result.json", result)
            results.append(result)

    csv_path, parquet_path = save_aggregate_results(results, output_dir)
    frame = results_to_dataframe(results)
    export_metric_matrix(
        frame,
        output_dir / "f1_matrix.csv",
        metric="f1",
        algorithms=selected_algorithms,
        complete_intersection=bool(stats_config.get("complete_intersection", False)),
    )

    logger.info("Saved CSV summary to %s", csv_path)
    if parquet_path is not None:
        logger.info("Saved parquet summary to %s", parquet_path)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DANTIS TSAD benchmark over UCR anomaly datasets.")
    parser.add_argument("--benchmark-config", default="experiment/config/benchmark_ucr.json", help="Benchmark configuration JSON.")
    parser.add_argument("--algorithms-config", default="experiment/config/algorithms.json", help="Algorithm registry configuration JSON.")
    parser.add_argument("--dataset-dir", default=None, help="Override the dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Override the output directory.")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Algorithms to run by name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the run.")
    parser.add_argument("--no-inclusive-end", action="store_true", help="Treat the anomaly end index as exclusive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_benchmark(
        args.benchmark_config,
        algorithms_config_path=args.algorithms_config,
        dataset_dir=args.dataset_dir,
        selected_algorithms=args.algorithms,
        output_dir=args.output_dir,
        random_seed=args.seed,
        anomaly_end_inclusive=not args.no_inclusive_end,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
