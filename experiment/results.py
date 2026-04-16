from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True, kw_only=True)
class BenchmarkResult:
    dataset: str
    collection: str
    algorithm: str
    hyperparameters_effective: dict[str, Any]
    fit_time: float
    predict_time: float
    precision: float | None
    recall: float | None
    f1: float | None
    average_precision: float | None
    roc_auc: float | None
    n_train: int
    n_test: int
    n_anomalies_test: int
    status: str
    score_source: str | None = None
    threshold_mode: str | None = None
    threshold_value: float | None = None
    output_mode_used: str | None = None
    algorithm_status: str | None = None
    input_mode_used: str | None = None
    error_message: str | None = None
    threshold: float | None = None
    y_pred_path: str | None = None
    y_score_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("metadata", None)
        return data


def save_numpy_array(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values))
    return str(path)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_result_json(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, sort_keys=True, default=_json_default)


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_record() for result in results])


def save_aggregate_results(results: list[BenchmarkResult], output_dir: str | Path) -> tuple[Path, Path | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = results_to_dataframe(results)
    csv_path = output_dir / "benchmark_results.csv"
    frame.to_csv(csv_path, index=False)

    parquet_path: Path | None = None
    try:
        parquet_path = output_dir / "benchmark_results.parquet"
        frame.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None

    return csv_path, parquet_path
