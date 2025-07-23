import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scienceplots  # noqa: F401
from tqdm import tqdm

from analysis import (
    compute_elo,
    compute_mle,
    compute_plackett_luce_pairwise_dense,
    compute_score_ranking,
    compute_tau,
    to_pairwise,
)
from analysis import load_results, load_filtered_vhelm_results

ROOT_DIR = Path(__file__).parents[2]


def plot_all_benchmarks(args):
    """Plot all benchmarks as subplots in a single figure with a shared legend."""
    plt.style.use(["science", "high-contrast", "grid"])
    plt.rcParams["font.family"] = "Times"
    benchmarks = ["helm", "leaderboard", "vhelm", "lmms-eval"]
    fig, axes = plt.subplots(1, 4, figsize=(args.fig_width * 4, args.fig_height * 1.2))

    for i, benchmark in enumerate(benchmarks):
        args.benchmark = benchmark
        taus = load_or_compute_taus(args)
        plot_taus(args, taus, benchmark, ax=axes[i])

    labels = ["Ours", "LMArena", "ELO"]
    handles, labels_unordered = axes[0].get_legend_handles_labels()
    handles = [handles[labels_unordered.index(label)] for label in labels]
    for handle in handles:
        handle.set_linewidth(2)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=14,
    )
    fig.tight_layout()

    path = args.fig_path / "sparsity.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def plot_single_benchmark(args):
    """Plot the robustness of the ranking to missing data for a single benchmark."""
    taus = load_or_compute_taus(args)
    fig, _ = plot_taus(args, taus, args.benchmark)

    path = args.fig_path / f"sparsity_{args.benchmark}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def load_or_compute_taus(args):
    """Test the robustness of the ranking to missing data."""
    output_path = Path(args.out_dir) / args.benchmark / "kendall_taus.csv"
    if output_path.exists():
        print("Loading existing data")
        return pl.read_csv(output_path)

    if args.benchmark == "vhelm":
        results = load_filtered_vhelm_results()
    else:
        results = load_results(args.benchmark)

    taus = compute_taus(results, args.benchmark)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    taus.write_csv(output_path)

    return taus


def filter_vhelm_results(results):
    """Filter out the rows with missing data from the VHELM results."""
    rows_to_drop = [5, 6, 12, 13, 14, 15, 16, 17, 19, 21]
    mask = ~pl.Series(range(results.height)).is_in(rows_to_drop)
    filtered_df = results.filter(mask)
    nulls_per_column = filtered_df.null_count().row(0)
    columns_with_no_nulls = [
        col
        for col, null_count in zip(results.columns, nulls_per_column)
        if null_count == 0
    ]
    return filtered_df.select(columns_with_no_nulls)


def compute_taus(results, benchmark):
    """Compute the Kendall's tau for the ranking of the models based on their average score."""
    ranking = compute_score_ranking(results, benchmark)

    taus_mle_mu = []
    taus_elo_mu = []
    taus_lsr_mu = []

    taus_mle_std = []
    taus_elo_std = []
    taus_lsr_std = []

    percentages = list(np.linspace(0, 0.9, 10)) + list(np.linspace(0.91, 0.99, 9))

    for fraction in tqdm(percentages):
        taus_mle = []
        taus_elo = []
        taus_lsr = []

        for i in range(3):
            random.seed(i * fraction)
            sampled = pl.from_pandas(
                results.to_pandas()
                .copy()
                .set_index("model")
                .apply(sparsify_row, axis=1, fraction=fraction)
                .reset_index()
            )
            pairwise_sampled = to_pairwise(sampled, long=False)

            ranking_mle = compute_mle(pairwise_sampled.to_pandas())
            ranking_elo = compute_elo(pairwise_sampled.to_pandas())
            ranking_lsr = compute_plackett_luce_pairwise_dense(pairwise_sampled)

            taus_mle.append(compute_tau(ranking.index, ranking_mle.index))
            taus_elo.append(compute_tau(ranking.index, ranking_elo.index))
            taus_lsr.append(compute_tau(ranking.index, ranking_lsr.index))

        taus_mle_mu.append(np.mean(taus_mle))
        taus_elo_mu.append(np.mean(taus_elo))
        taus_lsr_mu.append(np.mean(taus_lsr))

        taus_mle_std.append(np.std(taus_mle))
        taus_elo_std.append(np.std(taus_elo))
        taus_lsr_std.append(np.std(taus_lsr))

    return pl.DataFrame(
        {
            "percentage": percentages,
            "taus_lsr": taus_lsr_mu,
            "taus_lsr_std": taus_lsr_std,
            "taus_mle": taus_mle_mu,
            "taus_mle_std": taus_mle_std,
            "taus_elo": taus_elo_mu,
            "taus_elo_std": taus_elo_std,
        }
    )


def sparsify_row(row, fraction):
    """Sparsify a row by setting a fraction of each values to None."""
    n_replace = int(np.ceil(fraction * len(row)))
    indices_to_replace = random.sample(range(len(row)), n_replace)
    row.iloc[indices_to_replace] = None
    return row


def sparsify(df: pl.DataFrame, fraction: float) -> pl.DataFrame:
    """Sparsify a DataFrame by setting a fraction of the values to None."""
    mask = np.random.rand(df.height, df.width) > fraction
    model = pl.Series("model", df["model"])

    df = df.select(
        [
            pl.when(pl.lit(mask[:, i])).then(pl.col(col)).otherwise(None).alias(col)
            for i, col in enumerate(df.columns)
            if col != "model"
        ]
    )
    df = df.with_columns(model)

    return df


def plot_taus(args, data, benchmark, ax=None):
    """Plot the robustness of the ranking to missing data."""
    plt.style.use(["science", "high-contrast", "grid"])
    plt.rcParams["font.family"] = "Times"

    if ax is None:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    else:
        fig = None

    ax.plot(
        range(len(data["percentage"])),
        data["taus_mle"],
        label="LMArena",
    )
    ax.plot(
        range(len(data["percentage"])),
        data["taus_elo"],
        label="ELO",
    )
    ax.plot(
        range(len(data["percentage"])),
        data["taus_lsr"],
        label="Ours",
    )

    ax.fill_between(
        range(len(data["percentage"])),
        np.array(data["taus_mle"]) - np.array(data["taus_mle_std"]),
        np.array(data["taus_mle"]) + np.array(data["taus_mle_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        range(len(data["percentage"])),
        np.array(data["taus_elo"]) - np.array(data["taus_elo_std"]),
        np.array(data["taus_elo"]) + np.array(data["taus_elo_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        range(len(data["percentage"])),
        np.array(data["taus_lsr"]) - np.array(data["taus_lsr_std"]),
        np.array(data["taus_lsr"]) + np.array(data["taus_lsr_std"]),
        alpha=0.5,
    )

    # Set the x-axis ticks to the percentages and rotate them for better readability
    ax.set_xticks(range(len(data["percentage"])))
    ax.set_xticklabels([f"{p:.2f}" for p in data["percentage"]], rotation=45)

    # Set the y-axis limits and y-ticks
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ax.set_xticks(np.arange(0, len(data["percentage"]), 2))
    ax.set_xlim(0, len(data["percentage"]) - 1)
    ax.set_xlabel("Sparsity")
    if benchmark == "helm":
        ax.set_ylabel(r"Kendall's $\tau$ coefficient")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    benchmark_to_title = {
        "vhelm": "VHELM",
        "helm": "HELM",
        "leaderboard": "Open LLM Leaderboard",
        "lmms-eval": "LMMs-Eval",
    }
    ax.set_title(benchmark_to_title[benchmark], fontsize=15, pad=10)

    return fig, ax


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=ROOT_DIR / "results/sparsity/",
    )
    parser.add_argument(
        "--fig_path",
        type=Path,
        help="Where to save the figure",
        default=ROOT_DIR / "figures",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["vhelm", "helm", "leaderboard", "lmms-eval", "all"],
    )
    args = parser.parse_args()

    if args.benchmark == "all":
        plot_all_benchmarks(args)
    else:
        plot_single_benchmark(args)


if __name__ == "__main__":
    main()
