from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .stats import add_instance_id

DEFAULT_METRICS = ["f1", "precision", "recall", "average_precision", "roc_auc"]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_repeat_dirs(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.glob("repeat_*") if path.is_dir())


def _summarize(frame: pd.DataFrame, group_keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(group_keys, key_values)}

        for metric in metrics:
            valid = group[(group["status"] == "completed") & (group[metric].notna())]
            row[f"{metric}_mean"] = float(valid[metric].mean()) if not valid.empty else None
            row[f"{metric}_std"] = float(valid[metric].std(ddof=0)) if len(valid) > 1 else (0.0 if len(valid) == 1 else None)
            row[f"{metric}_count_valid"] = int(valid.shape[0])
            row[f"{metric}_count_failed"] = int(group.shape[0] - valid.shape[0])

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_experiments(
    output_dir: str | Path,
    *,
    metrics: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    repeat_dirs = _discover_repeat_dirs(output_dir)
    if not repeat_dirs:
        raise FileNotFoundError(f"No repeat directories found in {output_dir}")

    metrics = metrics or DEFAULT_METRICS

    frames: list[pd.DataFrame] = []
    for repeat_dir in repeat_dirs:
        csv_path = repeat_dir / "benchmark_results.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["repeat_id"] = repeat_dir.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No benchmark_results.csv files found in repeat directories.")

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs = add_instance_id(all_runs)

    aggregates_dir = output_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    all_runs_path = aggregates_dir / "all_runs_long.csv"
    all_runs.to_csv(all_runs_path, index=False)

    summary_by_algorithm_dataset = _summarize(
        all_runs,
        group_keys=["collection", "dataset", "instance_id", "algorithm"],
        metrics=[metric for metric in metrics if metric in all_runs.columns],
    )
    summary_by_algorithm_dataset_path = aggregates_dir / "summary_by_algorithm_dataset.csv"
    summary_by_algorithm_dataset.to_csv(summary_by_algorithm_dataset_path, index=False)

    summary_by_algorithm = _summarize(
        all_runs,
        group_keys=["algorithm"],
        metrics=[metric for metric in metrics if metric in all_runs.columns],
    )
    summary_by_algorithm_path = aggregates_dir / "summary_by_algorithm.csv"
    summary_by_algorithm.to_csv(summary_by_algorithm_path, index=False)

    return all_runs_path, summary_by_algorithm_dataset_path, summary_by_algorithm_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate repeated experiment outputs.")
    parser.add_argument("--output-dir", required=True, help="Root output directory containing repeat_* folders.")
    parser.add_argument("--experiment-config", default=None, help="Optional experiments config to read default metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    metrics = None
    if args.experiment_config:
        config = _load_json(args.experiment_config)
        metrics = config.get("metrics")

    aggregate_experiments(args.output_dir, metrics=metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
