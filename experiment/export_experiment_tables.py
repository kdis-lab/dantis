from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .stats import build_metric_matrix

DEFAULT_METRICS = ["f1", "precision", "recall", "average_precision", "roc_auc"]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export_experiment_tables(
    output_dir: str | Path,
    *,
    metrics: list[str] | None = None,
    algorithms: list[str] | None = None,
    complete_intersection: bool = False,
) -> list[Path]:
    output_dir = Path(output_dir)
    aggregates_dir = output_dir / "aggregates"
    input_path = aggregates_dir / "all_runs_long.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing aggregated long table: {input_path}")

    frame = pd.read_csv(input_path)
    metrics = metrics or DEFAULT_METRICS

    written_paths: list[Path] = []
    for metric in metrics:
        if metric not in frame.columns:
            continue

        valid = frame[(frame["status"] == "completed") & (frame[metric].notna())].copy()
        if valid.empty:
            continue

        mean_over_repeats = (
            valid.groupby(["instance_id", "algorithm"], as_index=False)[metric]
            .mean()
            .assign(status="completed")
        )

        matrix = build_metric_matrix(
            mean_over_repeats,
            metric=metric,
            algorithms=algorithms,
            complete_intersection=complete_intersection,
        )

        output_path = aggregates_dir / f"{metric}_matrix_mean.csv"
        matrix.to_csv(output_path)
        written_paths.append(output_path)

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-metric tables from aggregated repeated experiments.")
    parser.add_argument("--output-dir", required=True, help="Root output directory containing aggregates/all_runs_long.csv.")
    parser.add_argument("--experiment-config", default=None, help="Optional experiments config to read metrics/algorithms/stats.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    metrics = None
    algorithms = None
    complete_intersection = False

    if args.experiment_config:
        config = _load_json(args.experiment_config)
        metrics = config.get("metrics")
        algorithms = config.get("algorithms")
        stats_config = dict(config.get("stats", {}))
        complete_intersection = bool(stats_config.get("complete_intersection", False))

    export_experiment_tables(
        args.output_dir,
        metrics=metrics,
        algorithms=algorithms,
        complete_intersection=complete_intersection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
