from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import BenchmarkDataset


_UCR_FILENAME_PATTERN = re.compile(
    r"^(?P<dataset_id>\d+)_UCR_Anomaly_"
    r"(?P<dataset_name>.+)_"
    r"(?P<train_end>\d+)_"
    r"(?P<anomaly_start>\d+)_"
    r"(?P<anomaly_end>\d+)\.txt$"
)


@dataclass(slots=True, kw_only=True)
class UCRAnomalyFileSpec:
    """Metadata extracted from a UCR anomaly archive file name."""

    dataset_id: str
    dataset_name: str
    train_end: int
    anomaly_start: int
    anomaly_end: int
    source_file: str

    @property
    def anomaly_interval_inclusive(self) -> str:
        return "[anomaly_start, anomaly_end]"


@dataclass(slots=True, kw_only=True)
class UCRAnomalyDataset(BenchmarkDataset):
    """Concrete UCR anomaly dataset with full series reconstruction."""

    dataset_id: str
    source_file: str
    X_full: np.ndarray
    y_full: np.ndarray
    train_end: int
    anomaly_start: int
    anomaly_end: int
    anomaly_end_inclusive: bool = True

    @property
    def anomaly_start_test(self) -> int:
        return int(self.anomaly_start - self.train_end)

    @property
    def anomaly_end_test(self) -> int:
        return int(self.anomaly_end - self.train_end)

    def to_benchmark_dataset(self) -> BenchmarkDataset:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "dataset_id": self.dataset_id,
                "dataset_name": self.dataset_name,
                "source_file": self.source_file,
                "train_end": self.train_end,
                "anomaly_start": self.anomaly_start,
                "anomaly_end": self.anomaly_end,
                "n_samples": int(self.X_full.shape[0]),
                "n_features": int(self.X_full.shape[1]) if self.X_full.ndim == 2 else 1,
                "collection_name": self.collection_name,
            }
        )
        return BenchmarkDataset(
            dataset_name=self.dataset_name,
            collection_name=self.collection_name,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            metadata=metadata,
        )


def parse_ucr_anomaly_filename(file_path: str | Path) -> UCRAnomalyFileSpec:
    filename = Path(file_path).name
    match = _UCR_FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"Unexpected UCR anomaly filename format: {filename!r}. "
            "Expected something like '001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt'."
        )

    return UCRAnomalyFileSpec(
        dataset_id=match.group("dataset_id"),
        dataset_name=match.group("dataset_name"),
        train_end=int(match.group("train_end")),
        anomaly_start=int(match.group("anomaly_start")),
        anomaly_end=int(match.group("anomaly_end")),
        source_file=str(Path(file_path).resolve()),
    )


def _load_series(file_path: str | Path, dtype: type = np.float64) -> np.ndarray:
    series = np.loadtxt(file_path, dtype=dtype, ndmin=1)
    series = np.asarray(series, dtype=dtype)
    if series.ndim == 0:
        series = series.reshape(1)
    if series.ndim != 1:
        raise ValueError(
            f"Expected a univariate time series stored in a TXT file, got shape {series.shape}."
        )
    return series


def load_ucr_anomaly_file(
    file_path: str | Path,
    *,
    anomaly_end_inclusive: bool = True,
    dtype: type = np.float64,
) -> UCRAnomalyDataset:
    spec = parse_ucr_anomaly_filename(file_path)
    series = _load_series(file_path, dtype=dtype)

    n_samples = int(series.shape[0])
    train_end = int(spec.train_end)
    anomaly_start = int(spec.anomaly_start)
    anomaly_end = int(spec.anomaly_end)

    if n_samples < 2:
        raise ValueError(f"Series is too short: n_samples={n_samples}.")
    if not (0 < train_end < n_samples):
        raise ValueError(f"Invalid train_end={train_end} for series length n={n_samples}.")

    if anomaly_end_inclusive:
        valid_interval = train_end <= anomaly_start <= anomaly_end < n_samples
    else:
        valid_interval = train_end <= anomaly_start < anomaly_end <= n_samples

    if not valid_interval:
        raise ValueError(
            "Invalid anomaly interval parsed from filename: "
            f"start={anomaly_start}, end={anomaly_end}, n={n_samples}, train_end={train_end}, "
            f"anomaly_end_inclusive={anomaly_end_inclusive}."
        )

    X_full = series.reshape(-1, 1)
    y_full = np.zeros(n_samples, dtype=np.int32)
    if anomaly_end_inclusive:
        y_full[anomaly_start : anomaly_end + 1] = 1
    else:
        y_full[anomaly_start:anomaly_end] = 1

    X_train = X_full[:train_end]
    y_train = y_full[:train_end]
    X_test = X_full[train_end:]
    y_test = y_full[train_end:]

    metadata: dict[str, Any] = {
        "source_file": spec.source_file,
        "collection_name": "UCR_Anomaly",
        "dataset_id": spec.dataset_id,
        "dataset_name": spec.dataset_name,
        "train_end": train_end,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
        "n_samples": n_samples,
        "n_features": int(X_full.shape[1]),
    }

    return UCRAnomalyDataset(
        dataset_name=spec.dataset_name,
        collection_name="UCR_Anomaly",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata,
        dataset_id=spec.dataset_id,
        source_file=spec.source_file,
        X_full=X_full,
        y_full=y_full,
        train_end=train_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
        anomaly_end_inclusive=anomaly_end_inclusive,
    )


def load_ucr_anomaly_collection(
    collection_dir: str | Path,
    *,
    anomaly_end_inclusive: bool = True,
    dtype: type = np.float64,
) -> list[UCRAnomalyDataset]:
    directory = Path(collection_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    datasets: list[UCRAnomalyDataset] = []
    for file_path in sorted(directory.glob("*.txt")):
        datasets.append(
            load_ucr_anomaly_file(
                file_path,
                anomaly_end_inclusive=anomaly_end_inclusive,
                dtype=dtype,
            )
        )
    return datasets
