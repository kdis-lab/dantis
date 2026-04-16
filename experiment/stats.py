from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_instance_id(frame: pd.DataFrame, *, instance_column: str = "instance_id") -> pd.DataFrame:
    out = frame.copy()
    if instance_column in out.columns:
        return out
    if "collection" not in out.columns or "dataset" not in out.columns:
        raise KeyError("Results frame must contain 'collection' and 'dataset' columns to build instance_id.")
    out[instance_column] = out["collection"].astype(str) + "::" + out["dataset"].astype(str)
    return out


def filter_complete_instances(
    frame: pd.DataFrame,
    *,
    metric: str,
    algorithms: list[str],
    instance_column: str = "instance_id",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise KeyError(f"Metric {metric!r} not available in results frame.")

    out = add_instance_id(frame, instance_column=instance_column)
    completed = out[(out["status"] == "completed") & (out[metric].notna()) & (out["algorithm"].isin(algorithms))]
    counts = completed.groupby(instance_column)["algorithm"].nunique()
    valid_instances = counts[counts == len(set(algorithms))].index
    return completed[completed[instance_column].isin(valid_instances)]


def build_metric_matrix(
    frame: pd.DataFrame,
    metric: str = "f1",
    *,
    instance_column: str = "instance_id",
    algorithms: list[str] | None = None,
    complete_intersection: bool = False,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise KeyError(f"Metric {metric!r} not available in results frame.")

    out = add_instance_id(frame, instance_column=instance_column)
    if algorithms is not None:
        out = out[out["algorithm"].isin(algorithms)]

    if complete_intersection:
        if not algorithms:
            algorithms = sorted(out["algorithm"].dropna().unique().tolist())
        out = filter_complete_instances(out, metric=metric, algorithms=list(algorithms), instance_column=instance_column)

    return out.pivot_table(index=instance_column, columns="algorithm", values=metric, aggfunc="mean")


def export_metric_matrix(
    frame: pd.DataFrame,
    output_path: str | Path,
    metric: str = "f1",
    *,
    instance_column: str = "instance_id",
    algorithms: list[str] | None = None,
    complete_intersection: bool = False,
) -> Path:
    matrix = build_metric_matrix(
        frame,
        metric=metric,
        instance_column=instance_column,
        algorithms=algorithms,
        complete_intersection=complete_intersection,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)
    return output_path
