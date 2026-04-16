from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(slots=True, kw_only=True)
class AlgorithmSpec:
    """Descriptor for one benchmarked algorithm."""

    name: str
    module_path: str
    class_name: str
    default_hyperparameters: dict[str, Any] = field(default_factory=dict)
    init_mode: str = "auto"
    output_mode: str = "score"
    score_method_preference: tuple[str, ...] = ("decision_function", "get_anomaly_score", "score_samples")
    predict_returns_labels: bool = True
    allow_predict_as_score: bool = False
    requires_threshold: bool = True
    score_direction: str = "higher"
    input_mode: str = "auto"
    supported_data_type: str = "both"
    status: str = "experimental"
    series_types: tuple[str, ...] = ("UTS", "MTS")
    notes: str = ""

    def load_class(self):
        root_dir = Path(__file__).resolve().parents[1]
        module_path = Path(*self.module_path.split(".")).with_suffix(".py")
        file_path = root_dir / module_path

        importlib.import_module("dantis")
        self._ensure_namespace_package("dantis.machine_learning", root_dir / "dantis" / "machine_learning")
        self._ensure_namespace_package("dantis.statistical", root_dir / "dantis" / "statistical")
        self._ensure_namespace_package("dantis.deep_learning", root_dir / "dantis" / "deep_learning")

        spec = importlib.util.spec_from_file_location(self.module_path, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_path] = module
        spec.loader.exec_module(module)
        return getattr(module, self.class_name)

    @staticmethod
    def _ensure_namespace_package(package_name: str, package_path: Path) -> None:
        if package_name in sys.modules:
            return
        package = ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    def instantiate(self, hyperparameters: dict[str, Any] | None = None):
        cls = self.load_class()
        params = dict(self.default_hyperparameters)
        if hyperparameters:
            params.update(hyperparameters)

        signature = inspect.signature(cls.__init__)
        parameter_names = [name for name in signature.parameters if name != "self"]
        if self.init_mode == "kwargs":
            return cls(**params)
        if self.init_mode == "dict":
            if "hyperparameter" in parameter_names:
                return cls(hyperparameter=params)
            return cls(params)
        if "hyperparameter" in parameter_names:
            return cls(hyperparameter=params)
        return cls(**params)


ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "lof": AlgorithmSpec(
        name="lof",
        module_path="dantis.machine_learning.lof",
        class_name="LOF",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        score_direction="higher",
        input_mode="tabular_2d",
        supported_data_type="both",
        status="validated",
        notes="PyOD wrapper; predict() is score-like and thresholded in the benchmark layer.",
    ),
    "kmeans": AlgorithmSpec(
        name="kmeans",
        module_path="dantis.machine_learning.kmeans",
        class_name="KMeansAD",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        notes="Windowed KMeans implementation.",
    ),
    "dbstream": AlgorithmSpec(
        name="dbstream",
        module_path="dantis.machine_learning.dbstream",
        class_name="DBStreamAD",
        default_hyperparameters={},
        init_mode="kwargs",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        input_mode="tabular_2d",
        supported_data_type="both",
        status="experimental",
        notes="River-based streaming detector.",
    ),
    "ocsvm": AlgorithmSpec(
        name="ocsvm",
        module_path="dantis.machine_learning.ocsvm",
        class_name="OCSVM",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        score_direction="higher",
        input_mode="tabular_2d",
        supported_data_type="both",
        status="validated",
        notes="PyOD wrapper.",
    ),
    "rgraph": AlgorithmSpec(
        name="rgraph",
        module_path="dantis.machine_learning.rgraph",
        class_name="RGraph",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        input_mode="tabular_2d",
        supported_data_type="both",
        status="experimental",
        notes="PyOD wrapper.",
    ),
    "iforest": AlgorithmSpec(
        name="iforest",
        module_path="dantis.machine_learning.iforest",
        class_name="IForest",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        score_direction="higher",
        input_mode="tabular_2d",
        supported_data_type="both",
        status="validated",
        notes="PyOD wrapper.",
    ),
    "eif": AlgorithmSpec(
        name="eif",
        module_path="dantis.machine_learning.eif",
        class_name="EIFWrapper",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=False,
        requires_threshold=True,
        input_mode="tabular_2d",
        supported_data_type="both",
        status="experimental",
        notes="Depends on external eif package or local shim.",
    ),
    "grammarviz": AlgorithmSpec(
        name="grammarviz",
        module_path="dantis.machine_learning.grammarviz",
        class_name="GrammarViz",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        series_types=("UTS",),
        notes="GrammarViz-style symbolic detector for univariate series.",
    ),
    "hotsax": AlgorithmSpec(
        name="hotsax",
        module_path="dantis.statistical.hotsax",
        class_name="HOTSAX",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        series_types=("UTS",),
        notes="Discord discovery via saxpy.",
    ),
    "mp_damp": AlgorithmSpec(
        name="mp_damp",
        module_path="dantis.statistical.mp_damp",
        class_name="DAMP",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        notes="Matrix-profile based DAMP implementation.",
    ),
    "hbos": AlgorithmSpec(
        name="hbos",
        module_path="dantis.statistical.hbos",
        class_name="HBOS",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        score_direction="higher",
        input_mode="tabular_2d",
        supported_data_type="both",
        status="validated",
        notes="PyOD wrapper.",
    ),
    "ssa": AlgorithmSpec(
        name="ssa",
        module_path="dantis.statistical.ssa",
        class_name="SSA",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="validated",
        series_types=("UTS",),
        notes="Univariate SSA detector.",
    ),
    "arima": AlgorithmSpec(
        name="arima",
        module_path="dantis.statistical.arima",
        class_name="ARIMA",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="validated",
        series_types=("UTS",),
        notes="Statsmodels ARIMA anomaly wrapper.",
    ),
    "deepant": AlgorithmSpec(
        name="deepant",
        module_path="dantis.deep_learning.deepant",
        class_name="DeepAnT",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        notes="DeepAnT wrapper with torch backend.",
    ),
    "telemanom": AlgorithmSpec(
        name="telemanom",
        module_path="dantis.deep_learning.telemanom",
        class_name="TelemanomDetector",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="label",
        predict_returns_labels=True,
        requires_threshold=False,
        input_mode="series_1d",
        supported_data_type="UTS",
        status="experimental",
        notes="Keras/TensorFlow-based; heavy dependency surface.",
    ),
    "vae": AlgorithmSpec(
        name="vae",
        module_path="dantis.deep_learning.vae",
        class_name="VAE",
        default_hyperparameters={},
        init_mode="dict",
        output_mode="score",
        score_method_preference=("decision_function", "get_anomaly_score"),
        predict_returns_labels=True,
        requires_threshold=True,
        score_direction="higher",
        input_mode="tabular_2d",
        supported_data_type="both",
        status="validated",
        notes="PyOD VAE wrapper.",
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}") from exc


def create_algorithm(name: str, hyperparameters: dict[str, Any] | None = None):
    return get_algorithm_spec(name).instantiate(hyperparameters=hyperparameters)


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)
