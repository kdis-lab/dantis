from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment.aggregate_experiments import aggregate_experiments
from experiment.export_experiment_tables import export_experiment_tables
from experiment.run_experiments import run_experiments


class ExperimentOrchestrationTests(unittest.TestCase):
    def _write_common_configs(self, root: Path) -> tuple[Path, Path]:
        benchmark_config = root / "benchmark.json"
        algorithms_config = root / "algorithms.json"

        benchmark_config.write_text(
            json.dumps(
                {
                    "collection_name": "UCR_Anomaly",
                    "dataset_dir": str(root / "datasets"),
                    "output_dir": str(root / "outputs"),
                    "random_seed": 42,
                    "algorithms": ["lof"],
                }
            ),
            encoding="utf-8",
        )
        algorithms_config.write_text(
            json.dumps(
                {
                    "lof": {
                        "hyperparameters": {"n_neighbors": 5},
                        "threshold": {"mode": "contamination", "contamination": 0.1},
                        "score_direction": "higher",
                    }
                }
            ),
            encoding="utf-8",
        )
        return benchmark_config, algorithms_config

    def _write_dataset(self, dataset_dir: Path) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        values = [0, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        (dataset_dir / "001_UCR_Anomaly_demo_4_6_8.txt").write_text("\n".join(str(v) for v in values), encoding="utf-8")

    def _write_named_dataset(self, dataset_dir: Path, dataset_id: int, dataset_name: str) -> None:
        values = [0, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        filename = f"{dataset_id:03d}_UCR_Anomaly_{dataset_name}_4_6_8.txt"
        (dataset_dir / filename).write_text("\n".join(str(v) for v in values), encoding="utf-8")

    def test_run_experiments_creates_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            self._write_dataset(dataset_dir)
            benchmark_config, algorithms_config = self._write_common_configs(root)

            experiment_config = root / "experiment.json"
            experiment_config.write_text(
                json.dumps(
                    {
                        "benchmark_config": str(benchmark_config),
                        "algorithms_config": str(algorithms_config),
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(root / "outputs"),
                        "algorithms": ["lof"],
                        "n_repeats": 2,
                        "base_seed": 100,
                    }
                ),
                encoding="utf-8",
            )

            repeat_dirs = run_experiments(experiment_config)
            self.assertEqual(len(repeat_dirs), 2)
            self.assertTrue((root / "outputs" / "repeat_000" / "benchmark_results.csv").exists())
            self.assertTrue((root / "outputs" / "repeat_001" / "benchmark_results.csv").exists())

    def test_aggregate_and_export_generates_outputs_and_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            repeat0 = output_dir / "repeat_000"
            repeat1 = output_dir / "repeat_001"
            repeat0.mkdir(parents=True)
            repeat1.mkdir(parents=True)

            frame0 = pd.DataFrame(
                [
                    {
                        "collection": "UCR_Anomaly",
                        "dataset": "demo",
                        "algorithm": "lof",
                        "status": "completed",
                        "f1": 0.2,
                        "precision": 0.2,
                        "recall": 0.2,
                    }
                ]
            )
            frame1 = pd.DataFrame(
                [
                    {
                        "collection": "UCR_Anomaly",
                        "dataset": "demo",
                        "algorithm": "lof",
                        "status": "completed",
                        "f1": 0.8,
                        "precision": 0.8,
                        "recall": 0.8,
                    }
                ]
            )
            frame0.to_csv(repeat0 / "benchmark_results.csv", index=False)
            frame1.to_csv(repeat1 / "benchmark_results.csv", index=False)

            all_runs_path, _, _ = aggregate_experiments(output_dir, metrics=["f1", "precision", "recall"])
            self.assertTrue(all_runs_path.exists())

            written = export_experiment_tables(
                output_dir,
                metrics=["f1"],
                algorithms=["lof"],
                complete_intersection=True,
            )
            self.assertTrue(written)

            matrix = pd.read_csv(output_dir / "aggregates" / "f1_matrix_mean.csv", index_col=0)
            self.assertAlmostEqual(float(matrix.loc["UCR_Anomaly::demo", "lof"]), 0.5, places=6)

    def test_max_datasets_limits_final_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self._write_named_dataset(dataset_dir, 1, "alpha")
            self._write_named_dataset(dataset_dir, 2, "beta")
            self._write_named_dataset(dataset_dir, 3, "gamma")

            benchmark_config, algorithms_config = self._write_common_configs(root)
            experiment_config = root / "experiment.json"
            experiment_config.write_text(
                json.dumps(
                    {
                        "benchmark_config": str(benchmark_config),
                        "algorithms_config": str(algorithms_config),
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(root / "outputs"),
                        "algorithms": ["lof"],
                        "n_repeats": 1,
                    }
                ),
                encoding="utf-8",
            )

            run_experiments(experiment_config, max_datasets=2)
            frame = pd.read_csv(root / "outputs" / "repeat_000" / "benchmark_results.csv")
            self.assertEqual(frame["dataset"].nunique(), 2)

    def test_include_exclude_and_list_datasets_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            self._write_named_dataset(dataset_dir, 1, "alpha")
            self._write_named_dataset(dataset_dir, 2, "beta")
            self._write_named_dataset(dataset_dir, 3, "gamma")

            benchmark_config, algorithms_config = self._write_common_configs(root)
            experiment_config = root / "experiment.json"
            experiment_config.write_text(
                json.dumps(
                    {
                        "benchmark_config": str(benchmark_config),
                        "algorithms_config": str(algorithms_config),
                        "dataset_dir": str(dataset_dir),
                        "output_dir": str(root / "outputs"),
                        "algorithms": ["lof"],
                        "include_datasets": ["alpha", "beta"],
                        "exclude_datasets": ["beta"],
                        "n_repeats": 1,
                    }
                ),
                encoding="utf-8",
            )

            listed = run_experiments(experiment_config, list_datasets=True, verbose=True)
            self.assertEqual(listed, [])
            self.assertFalse((root / "outputs" / "repeat_000").exists())

            run_experiments(experiment_config)
            frame = pd.read_csv(root / "outputs" / "repeat_000" / "benchmark_results.csv")
            self.assertEqual(frame["dataset"].nunique(), 1)
            self.assertEqual(frame["dataset"].iloc[0], "alpha")


if __name__ == "__main__":
    unittest.main()
