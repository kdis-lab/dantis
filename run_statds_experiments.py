
import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from statds.no_parametrics import friedman, nemenyi


DEFAULT_METRICS = [
    "f1",
    "precision",
    "recall",
    "average_precision",
    "roc_auc",
]


def load_matrix(csv_path: Path) -> pd.DataFrame:
    """
    Load a metric matrix in the format required by StaTDS.

    Expected format:
    - first column: dataset / instance identifier
    - remaining columns: one column per algorithm
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] < 3:
        raise ValueError(
            f"{csv_path} must contain at least 3 columns: "
            "dataset identifier + at least 2 algorithm columns."
        )

    return df


def save_markdown_report(
    report_path: Path,
    metric_name: str,
    df: pd.DataFrame,
    alpha: float,
    hypothesis: str,
    statistic: float,
    critical_value: float,
    p_value: float,
    rankings,
    ranks_values,
    critical_distance_nemenyi,
    figure_path: Path | None,
) -> None:
    """Save a Markdown report with Friedman and Nemenyi results."""
    md = []
    md.append(f"# Statistical analysis for metric: `{metric_name}`\n")
    md.append(f"- Input file: `{df.attrs.get('source_file', 'unknown')}`")
    md.append(f"- Number of instances: **{df.shape[0]}**")
    md.append(f"- Number of algorithms: **{df.shape[1] - 1}**")
    md.append(f"- Alpha: **{alpha}**\n")

    md.append("## Friedman test\n")
    md.append(f"- Hypothesis: **{hypothesis}**")
    md.append(f"- Statistic: **{statistic}**")
    md.append(f"- Critical value: **{critical_value}**")
    md.append(f"- p-value: **{p_value}**\n")

    md.append("## Average rankings\n")
    try:
        rankings_df = pd.DataFrame(rankings)
        md.append(rankings_df.to_markdown(index=False))
    except Exception:
        md.append("```")
        md.append(str(rankings))
        md.append("```")
    md.append("")

    md.append("## Nemenyi post-hoc\n")
    md.append(f"- Critical distance: **{critical_distance_nemenyi}**\n")

    try:
        ranks_values_df = pd.DataFrame(ranks_values)
        md.append(ranks_values_df.to_markdown(index=False))
    except Exception:
        md.append("```")
        md.append(str(ranks_values))
        md.append("```")
    md.append("")

    if figure_path is not None:
        md.append("## Critical difference diagram\n")
        md.append(f"![Nemenyi diagram]({figure_path.name})\n")

    report_path.write_text("\n".join(md), encoding="utf-8")


def analyse_metric(csv_path: Path, output_dir: Path, alpha: float, minimize: bool) -> None:
    """Run Friedman + Nemenyi for one metric file and save results."""
    metric_name = csv_path.name.replace("_matrix_mean.csv", "").replace(".csv", "")
    metric_out_dir = output_dir / metric_name
    metric_out_dir.mkdir(parents=True, exist_ok=True)

    df = load_matrix(csv_path)
    df.attrs["source_file"] = str(csv_path)

    rankings, statistic, p_value, critical_value, hypothesis = friedman(
        df, alpha, minimize=minimize
    )

    num_cases = df.shape[0]
    ranks_values, critical_distance_nemenyi, figure = nemenyi(rankings, num_cases, alpha)

    try:
        pd.DataFrame(rankings).to_csv(metric_out_dir / "friedman_rankings.csv", index=False)
    except Exception:
        (metric_out_dir / "friedman_rankings.txt").write_text(str(rankings), encoding="utf-8")

    try:
        pd.DataFrame(ranks_values).to_csv(metric_out_dir / "nemenyi_ranks.csv", index=False)
    except Exception:
        (metric_out_dir / "nemenyi_ranks.txt").write_text(str(ranks_values), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "alpha": alpha,
                "num_cases": num_cases,
                "num_algorithms": df.shape[1] - 1,
                "hypothesis": hypothesis,
                "statistic": statistic,
                "critical_value": critical_value,
                "p_value": p_value,
                "critical_distance_nemenyi": critical_distance_nemenyi,
                "minimize": minimize,
                "source_file": str(csv_path),
            }
        ]
    )
    summary.to_csv(metric_out_dir / "summary.csv", index=False)

    figure_path = None
    try:
        figure_path = metric_out_dir / f"{metric_name}_nemenyi.png"
        figure.savefig(figure_path, bbox_inches="tight", dpi=200)
    except Exception:
        figure_path = None

    save_markdown_report(
        report_path=metric_out_dir / "report.md",
        metric_name=metric_name,
        df=df,
        alpha=alpha,
        hypothesis=hypothesis,
        statistic=statistic,
        critical_value=critical_value,
        p_value=p_value,
        rankings=rankings,
        ranks_values=ranks_values,
        critical_distance_nemenyi=critical_distance_nemenyi,
        figure_path=figure_path,
    )


def discover_metric_files(input_dir: Path, metrics: Iterable[str]) -> list[Path]:
    """Discover metric matrix files under an input directory."""
    files = []
    for metric in metrics:
        candidates = [
            input_dir / f"{metric}_matrix_mean.csv",
            input_dir / "metrics" / f"{metric}_matrix_mean.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                files.append(candidate)
                break
    return files


def build_combined_report(output_dir: Path) -> None:
    """Build a global Markdown index for all metric reports."""
    metric_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir()])
    lines = ["# Statistical analysis summary\n"]

    for metric_dir in metric_dirs:
        summary_csv = metric_dir / "summary.csv"
        report_md = metric_dir / "report.md"
        if not summary_csv.exists():
            continue

        summary = pd.read_csv(summary_csv).iloc[0]
        lines.append(f"## `{metric_dir.name}`\n")
        lines.append(f"- Hypothesis: **{summary['hypothesis']}**")
        lines.append(f"- Statistic: **{summary['statistic']}**")
        lines.append(f"- Critical value: **{summary['critical_value']}**")
        lines.append(f"- p-value: **{summary['p_value']}**")
        lines.append(f"- Critical distance: **{summary['critical_distance_nemenyi']}**")
        if report_md.exists():
            lines.append(f"- Detailed report: `{metric_dir.name}/report.md`")
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Friedman + Nemenyi statistical tests with StaTDS for each metric matrix."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing metric matrices, e.g. experiment/results/.../aggregates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where statistical test outputs will be stored.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to analyse. Expected file names: <metric>_matrix_mean.csv",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the tests.",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Use minimize=True in StaTDS. Leave unset when higher metric values are better.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = discover_metric_files(args.input_dir, args.metrics)

    if not metric_files:
        raise FileNotFoundError(
            f"No metric matrices found in {args.input_dir} for metrics {args.metrics}."
        )

    for csv_path in metric_files:
        print(f"[INFO] Analysing {csv_path.name}")
        analyse_metric(
            csv_path=csv_path,
            output_dir=args.output_dir,
            alpha=args.alpha,
            minimize=args.minimize,
        )

    build_combined_report(args.output_dir)
    print(f"[INFO] Statistical reports saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

