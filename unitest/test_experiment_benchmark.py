from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.registry import ALGORITHM_REGISTRY, AlgorithmSpec
from experiment.results import BenchmarkResult, save_result_json
from experiment.run_benchmark import run_benchmark
from experiment.stats import build_metric_matrix
from experiment.ucr_loader import load_ucr_anomaly_file, parse_ucr_anomaly_filename


class DummyLabelDetector:
    def __init__(self, hyperparameter=None, **kwargs):
        self._threshold = 0.0

    def fit(self, X, y=None):
        train_values = np.asarray(X, dtype=float).reshape(-1)
        self._threshold = float(np.median(train_values))
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float).reshape(-1)
        return (values >= self._threshold).astype(int)


class DummyScoreDetector:
    def __init__(self, hyperparameter=None, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float).reshape(-1)


class UCRLoaderTests(unittest.TestCase):
    def test_parse_ucr_filename(self):
        spec = parse_ucr_anomaly_filename(
            "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
        )
        self.assertEqual(spec.dataset_id, "001")
        self.assertEqual(spec.dataset_name, "DISTORTED1sddb40")
        self.assertEqual(spec.train_end, 35000)
        self.assertEqual(spec.anomaly_start, 52000)
        self.assertEqual(spec.anomaly_end, 52620)

    def test_load_ucr_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "001_UCR_Anomaly_demo_4_6_8.txt"
            np.savetxt(file_path, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float))

            dataset = load_ucr_anomaly_file(file_path)
            self.assertEqual(dataset.X_full.shape, (10, 1))
            self.assertEqual(dataset.X_train.shape, (4, 1))
            self.assertEqual(dataset.X_test.shape, (6, 1))
            self.assertEqual(int(dataset.y_full.sum()), 3)
            self.assertEqual(int(dataset.y_test.sum()), 3)
            self.assertEqual(dataset.anomaly_start_test, 2)
            self.assertEqual(dataset.anomaly_end_test, 4)


class ResultSerializationTests(unittest.TestCase):
    def test_result_json_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result = BenchmarkResult(
                dataset="demo",
                collection="UCR_Anomaly",
                algorithm="lof",
                hyperparameters_effective={"n_neighbors": 5},
                fit_time=0.1,
                predict_time=0.2,
                precision=0.5,
                recall=0.5,
                f1=0.5,
                average_precision=0.5,
                roc_auc=0.5,
                n_train=4,
                n_test=6,
                n_anomalies_test=3,
                status="completed",
                threshold=0.7,
                y_pred_path="/tmp/y_pred.npy",
                y_score_path="/tmp/y_score.npy",
            )
            save_result_json(path, result)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "demo")
            self.assertEqual(payload["algorithm"], "lof")
            self.assertEqual(payload["status"], "completed")


class RunnerSmokeTests(unittest.TestCase):
    def test_small_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = tmpdir / "UCR_Anomaly"
            output_dir = tmpdir / "outputs"
            dataset_dir.mkdir()

            series = np.array([0, 1, 0, 1, 0, 5, 6, 7, 8, 9], dtype=float)
            file_path = dataset_dir / "001_UCR_Anomaly_demo_4_6_8.txt"
            np.savetxt(file_path, series)

            benchmark_config = tmpdir / "benchmark.json"
            algorithms_config = tmpdir / "algorithms.json"

            benchmark_config.write_text(
                json.dumps(
                    {
                        "collection_name": "UCR_Anomaly",
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(output_dir),
                        "random_seed": 42,
                        "algorithms": ["lof"],
                        "threshold": {"mode": "contamination", "contamination": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            algorithms_config.write_text(
                json.dumps(
                    {
                        "lof": {
                            "hyperparameters": {},
                            "threshold": {"mode": "contamination", "contamination": 0.1},
                            "score_direction": "higher",
                        }
                    }
                ),
                encoding="utf-8",
            )

            results = run_benchmark(
                benchmark_config,
                algorithms_config_path=algorithms_config,
                dataset_dir=dataset_dir,
                selected_algorithms=["lof"],
                output_dir=output_dir,
                random_seed=42,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "completed")
            self.assertTrue((output_dir / "benchmark_results.csv").exists())
            self.assertTrue((output_dir / "f1_matrix.csv").exists())


class RunnerOutputModeTests(unittest.TestCase):
    def test_label_mode_skips_threshold_and_records_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = tmpdir / "UCR_Anomaly"
            output_dir = tmpdir / "outputs"
            dataset_dir.mkdir()

            series = np.array([0, 1, 0, 1, 0, 5, 6, 7, 8, 9], dtype=float)
            file_path = dataset_dir / "001_UCR_Anomaly_demo_label_4_6_8.txt"
            np.savetxt(file_path, series)

            benchmark_config = tmpdir / "benchmark.json"
            algorithms_config = tmpdir / "algorithms.json"

            benchmark_config.write_text(
                json.dumps(
                    {
                        "collection_name": "UCR_Anomaly",
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(output_dir),
                        "random_seed": 42,
                        "algorithms": ["dummy_label"],
                    }
                ),
                encoding="utf-8",
            )
            algorithms_config.write_text(json.dumps({"dummy_label": {"hyperparameters": {}}}), encoding="utf-8")

            ALGORITHM_REGISTRY["dummy_label"] = AlgorithmSpec(
                name="dummy_label",
                module_path="unitest.test_experiment_benchmark",
                class_name="DummyLabelDetector",
                init_mode="dict",
                output_mode="label",
                predict_returns_labels=True,
                requires_threshold=False,
                status="validated",
                input_mode="series_1d",
                supported_data_type="UTS",
            )

            try:
                results = run_benchmark(
                    benchmark_config,
                    algorithms_config_path=algorithms_config,
                    dataset_dir=dataset_dir,
                    selected_algorithms=["dummy_label"],
                    output_dir=output_dir,
                    random_seed=42,
                )
            finally:
                ALGORITHM_REGISTRY.pop("dummy_label", None)

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.output_mode_used, "label")
            self.assertEqual(result.score_source, "predict_binary")
            self.assertIsNone(result.threshold_mode)
            self.assertIsNone(result.threshold_value)
            self.assertTrue(Path(result.y_pred_path).exists())
            self.assertIsNone(result.y_score_path)

    def test_score_mode_records_threshold_and_score_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = tmpdir / "UCR_Anomaly"
            output_dir = tmpdir / "outputs"
            dataset_dir.mkdir()

            series = np.array([0, 1, 0, 1, 0, 5, 6, 7, 8, 9], dtype=float)
            file_path = dataset_dir / "001_UCR_Anomaly_demo_score_4_6_8.txt"
            np.savetxt(file_path, series)

            benchmark_config = tmpdir / "benchmark.json"
            algorithms_config = tmpdir / "algorithms.json"

            benchmark_config.write_text(
                json.dumps(
                    {
                        "collection_name": "UCR_Anomaly",
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(output_dir),
                        "random_seed": 42,
                        "algorithms": ["dummy_score"],
                    }
                ),
                encoding="utf-8",
            )
            algorithms_config.write_text(
                json.dumps(
                    {
                        "dummy_score": {
                            "hyperparameters": {},
                            "threshold": {"mode": "percentile", "percentile": 80.0},
                            "score_direction": "higher",
                        }
                    }
                ),
                encoding="utf-8",
            )

            ALGORITHM_REGISTRY["dummy_score"] = AlgorithmSpec(
                name="dummy_score",
                module_path="unitest.test_experiment_benchmark",
                class_name="DummyScoreDetector",
                init_mode="dict",
                output_mode="score",
                score_method_preference=("decision_function",),
                predict_returns_labels=False,
                requires_threshold=True,
                status="validated",
                input_mode="tabular_2d",
                supported_data_type="both",
            )

            try:
                results = run_benchmark(
                    benchmark_config,
                    algorithms_config_path=algorithms_config,
                    dataset_dir=dataset_dir,
                    selected_algorithms=["dummy_score"],
                    output_dir=output_dir,
                    random_seed=42,
                )
            finally:
                ALGORITHM_REGISTRY.pop("dummy_score", None)

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.output_mode_used, "score")
            self.assertEqual(result.score_source, "decision_function")
            self.assertEqual(result.threshold_mode, "percentile")
            self.assertIsNotNone(result.threshold_value)
            self.assertTrue(Path(result.y_pred_path).exists())
            self.assertTrue(Path(result.y_score_path).exists())


class StatsTests(unittest.TestCase):
    def test_metric_matrix_uses_robust_instance_id_and_intersection(self):
        frame = pd.DataFrame(
            [
                {"collection": "UCR_Anomaly", "dataset": "A", "algorithm": "alg1", "status": "completed", "f1": 0.8},
                {"collection": "UCR_Anomaly", "dataset": "A", "algorithm": "alg2", "status": "completed", "f1": 0.6},
                {"collection": "UCR_Anomaly", "dataset": "B", "algorithm": "alg1", "status": "completed", "f1": 0.7},
                {"collection": "UCR_Anomaly", "dataset": "B", "algorithm": "alg2", "status": "failed", "f1": None},
            ]
        )

        matrix = build_metric_matrix(
            frame,
            metric="f1",
            algorithms=["alg1", "alg2"],
            complete_intersection=True,
        )

        self.assertIn("UCR_Anomaly::A", matrix.index)
        self.assertNotIn("UCR_Anomaly::B", matrix.index)


if __name__ == "__main__":
    unittest.main()
