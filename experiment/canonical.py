from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, kw_only=True)
class BenchmarkDataset:
    """Canonical dataset representation for benchmark execution."""

    dataset_name: str
    collection_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1]) if self.X_train.ndim == 2 else 0

    @property
    def n_anomalies_test(self) -> int:
        return int(np.asarray(self.y_test).sum())
