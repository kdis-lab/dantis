from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .run_benchmark import run_benchmark
from .ucr_loader import parse_ucr_anomaly_filename


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_dataset_files(
    dataset_dir: Path,
    *,
    include_datasets: list[str] | None,
    exclude_datasets: list[str] | None,
) -> tuple[list[Path], dict[str, int]]:
    include = set(include_datasets or [])
    exclude = set(exclude_datasets or [])

    all_files = sorted(dataset_dir.glob("*.txt"))
    total_count = len(all_files)

    included: list[Path] = []
    for file_path in all_files:
        spec = parse_ucr_anomaly_filename(file_path)
        name = spec.dataset_name
        if include and name not in include:
            continue
        included.append(file_path)

    selected: list[Path] = []
    for file_path in included:
        spec = parse_ucr_anomaly_filename(file_path)
        name = spec.dataset_name
        if name in exclude:
            continue
        selected.append(file_path)

    stats = {
        "total": total_count,
        "after_include": len(included),
        "after_exclude": len(selected),
    }
    return selected, stats


def _build_selected_dataset_dir(output_dir: Path, dataset_files: list[Path]) -> Path:
    selected_dir = output_dir / "_selected_datasets"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    for source in dataset_files:
        target = selected_dir / source.name
        try:
            target.symlink_to(source.resolve())
        except Exception:
            shutil.copy2(source, target)

    return selected_dir


def run_experiments(
    experiment_config_path: str | Path,
    *,
    max_datasets: int | None = None,
    list_datasets: bool = False,
    verbose: bool = False,
) -> list[Path]:
    config = _load_json(experiment_config_path)

    benchmark_config_path = Path(config.get("benchmark_config", "experiment/config/benchmark_ucr.json"))
    algorithms_config_path = Path(config.get("algorithms_config", "experiment/config/algorithms.json"))

    benchmark_config = _load_json(benchmark_config_path)

    dataset_dir = Path(config.get("dataset_dir", benchmark_config.get("dataset_dir", "experiment/datasets/UCR_Anomaly")))
    output_dir = Path(config.get("output_dir", "experiment/results/ucr_experiment_v1"))
    algorithms = config.get("algorithms", benchmark_config.get("algorithms", []))
    include_datasets = config.get("include_datasets")
    exclude_datasets = config.get("exclude_datasets")
    n_repeats = int(config.get("n_repeats", 1))
    base_seed = int(config.get("base_seed", benchmark_config.get("random_seed", 42)))
    anomaly_end_inclusive = bool(config.get("anomaly_end_inclusive", True))
    max_datasets = max_datasets if max_datasets is not None else config.get("max_datasets")

    print("[INFO] Starting experiments")
    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] Selected algorithms: {algorithms}")

    dataset_files, selection_stats = _select_dataset_files(
        dataset_dir,
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
    )

    if verbose:
        print(f"[INFO] Datasets total={selection_stats['total']}")
        print(f"[INFO] Datasets after include={selection_stats['after_include']}")
        print(f"[INFO] Datasets after exclude={selection_stats['after_exclude']}")

    if max_datasets is not None:
        max_datasets = int(max_datasets)
        dataset_files = sorted(dataset_files, key=lambda path: path.name)[:max_datasets]
        if verbose:
            print(f"[INFO] Datasets after max_datasets={len(dataset_files)}")

    if not dataset_files:
        raise ValueError("No datasets selected after include/exclude filtering.")

    dataset_names = [parse_ucr_anomaly_filename(path).dataset_name for path in dataset_files]
    print(f"[INFO] Selected datasets: {len(dataset_files)}")
    if list_datasets or len(dataset_files) <= 10:
        for index, dataset_name in enumerate(dataset_names, start=1):
            print(f"[INFO] dataset[{index:03d}]={dataset_name}")

    if list_datasets:
        print("[INFO] list-datasets requested, exiting before execution")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dataset_dir = _build_selected_dataset_dir(output_dir, dataset_files)

    config_snapshot = dict(config)
    config_snapshot.setdefault("collection_name", benchmark_config.get("collection_name", "UCR_Anomaly"))
    config_snapshot["selected_dataset_files"] = [path.name for path in dataset_files]
    config_snapshot["selected_dataset_count"] = len(dataset_files)
    config_snapshot["benchmark_config"] = str(benchmark_config_path)
    config_snapshot["algorithms_config"] = str(algorithms_config_path)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    written_repeat_dirs: list[Path] = []
    for repeat_id in range(n_repeats):
        repeat_seed = base_seed + repeat_id
        repeat_number = repeat_id + 1
        repeat_dir = output_dir / f"repeat_{repeat_id:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Starting repeat {repeat_number}/{n_repeats} with seed={repeat_seed}")

        run_benchmark(
            benchmark_config_path,
            algorithms_config_path=algorithms_config_path,
            dataset_dir=selected_dataset_dir,
            selected_algorithms=algorithms,
            output_dir=repeat_dir,
            random_seed=repeat_seed,
            anomaly_end_inclusive=anomaly_end_inclusive,
        )

        (repeat_dir / "repeat_metadata.json").write_text(
            json.dumps(
                {
                    "repeat_id": repeat_id,
                    "seed": repeat_seed,
                    "algorithms": algorithms,
                    "dataset_count": len(dataset_files),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        written_repeat_dirs.append(repeat_dir)
        print(f"[INFO] Finished repeat {repeat_number}/{n_repeats}")

    print("[INFO] Finished experiments")

    return written_repeat_dirs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated TSAD experiments using the existing benchmark pipeline.")
    parser.add_argument(
        "--experiment-config",
        default="experiment/config/experiment_ucr_example.json",
        help="Path to the experiments JSON configuration.",
    )
    parser.add_argument("--max-datasets", type=int, default=None, help="Limit final selected datasets after include/exclude filtering.")
    parser.add_argument("--list-datasets", action="store_true", help="List selected datasets and exit without running repeats.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed dataset selection counts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_experiments(
        args.experiment_config,
        max_datasets=args.max_datasets,
        list_datasets=args.list_datasets,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
