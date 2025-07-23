import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scienceplots  # noqa: F401
from tqdm import tqdm

from analysis import (
    compute_plackett_luce_pairwise_dense,
    compute_score_ranking,
    compute_tau,
    to_pairwise,
)
from analysis import (
    load_results,
    load_synthetic_model_parameters,
)

ROOT_DIR = Path(__file__).parents[2]


def plot_all_benchmarks(args):
    """Plot all benchmarks as subplots in a single figure with a shared legend."""
    plt.style.use(["science", "high-contrast", "grid"])
    plt.rcParams["font.family"] = "Times"
    benchmarks = ["helm", "leaderboard", "vhelm", "lmms-eval"]
    fig, axes = plt.subplots(
        1, 4, figsize=(args.fig_width * len(benchmarks), args.fig_height * 1.2)
    )

    for i, benchmark in enumerate(benchmarks):
        args.benchmark = benchmark
        df = load_or_compute_taus(args)
        plot_taus(args, df, ax=axes[i])

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
    path = args.fig_path / f"robustness.{args.format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def plot_single_benchmark(args):
    """Plot the robustness of the ranking to missing data for a single benchmark."""
    df = load_or_compute_taus(args)
    fig, _ = plot_taus(args, df)

    path = args.fig_path / f"robustness_{args.benchmark}.{args.format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def load_or_compute_taus(args):
    """Test the robustness of the ranking to missing data."""
    output_path = Path(args.out_dir) / args.benchmark / "kendall_taus.csv"
    if output_path.exists() and not args.overwrite:
        print("Loading existing data")
        return pl.read_csv(output_path)

    # if args.benchmark == "vhelm":
    #     results = load_filtered_vhelm_results()
    # else:
    #     
    results = load_results(args.benchmark)

    if args.benchmark == "synthetic":
        ranking = load_synthetic_model_parameters()
    else:
        ranking = compute_score_ranking(results, args.benchmark)

    taus = compute_taus(results, ranking, args.percentages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    taus.write_csv(output_path)

    return taus


def compute_taus(results, global_ranking, percentages=None):
    """Compute the Kendall Tau correlation of the ranking with missing data."""
    taus_mle_mu = []
    taus_elo_mu = []
    taus_lsr_mu = []

    taus_mle_std = []
    taus_elo_std = []
    taus_lsr_std = []

    if percentages is None:
        percentages = list(np.linspace(0, 0.9, 10)) + list(np.linspace(0.91, 0.99, 9))

    for fraction in tqdm(percentages):
        taus_mle = []
        taus_elo = []
        taus_lsr = []

        for _ in range(1):
            columns = results.columns
            sampled_columns = random.sample(columns, int((1 - fraction) * len(columns)))
            sampled = results.select(sampled_columns)
            if "model" not in sampled.columns:
                sampled = sampled.with_columns(results["model"].alias("model"))
            pairwise_sampled = to_pairwise(sampled, long=False)

            # ranking_mle = compute_mle(pairwise_sampled.to_pandas())
            # ranking_elo = compute_elo(pairwise_sampled.to_pandas())
            ranking_lsr = compute_plackett_luce_pairwise_dense(pairwise_sampled)
            print(ranking_lsr)
            break

            taus_mle.append(compute_tau(global_ranking.index, ranking_mle.index))
            taus_elo.append(compute_tau(global_ranking.index, ranking_elo.index))
            taus_lsr.append(compute_tau(global_ranking.index, ranking_lsr.index))

        taus_mle_mu.append(np.mean(taus_mle))
        taus_elo_mu.append(np.mean(taus_elo))
        taus_lsr_mu.append(np.mean(taus_lsr))

        taus_mle_std.append(np.std(taus_mle))
        taus_elo_std.append(np.std(taus_elo))
        taus_lsr_std.append(np.std(taus_lsr))
        print(taus_lsr_mu)
    return pl.DataFrame(
        {
            "percentage": percentages,
            "taus_mle": taus_mle_mu,
            "taus_elo": taus_elo_mu,
            "taus_mle_std": taus_mle_std,
            "taus_elo_std": taus_elo_std,
            "taus_lsr": taus_lsr_mu,
            "taus_lsr_std": taus_lsr_std,
        }
    )


def plot_taus(args, df, ax=None):
    """Plot the robustness of the ranking to missing data."""
    plt.style.use(["science", "high-contrast", "grid"])
    plt.rcParams["font.family"] = "Times"

    if ax is None:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        is_subplot = False
    else:
        fig = None
        is_subplot = True

    ax.plot(
        range(len(df["percentage"])),
        df["taus_mle"],
        label="LMArena",
    )
    ax.plot(
        range(len(df["percentage"])),
        df["taus_elo"],
        label="ELO",
    )
    ax.plot(
        range(len(df["percentage"])),
        df["taus_lsr"],
        label="Ours",
    )

    ax.fill_between(
        range(len(df["percentage"])),
        np.array(df["taus_mle"]) - np.array(df["taus_mle_std"]),
        np.array(df["taus_mle"]) + np.array(df["taus_mle_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        range(len(df["percentage"])),
        np.array(df["taus_elo"]) - np.array(df["taus_elo_std"]),
        np.array(df["taus_elo"]) + np.array(df["taus_elo_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        range(len(df["percentage"])),
        np.array(df["taus_lsr"]) - np.array(df["taus_lsr_std"]),
        np.array(df["taus_lsr"]) + np.array(df["taus_lsr_std"]),
        alpha=0.5,
    )

    # Set the x-axis ticks to the percentages and rotate them for better readability
    ax.set_xticks(range(len(df["percentage"])))
    ax.set_xticklabels([f"{p:.2f}" for p in df["percentage"]], rotation=45)

    # Set the y-axis limits and y-ticks
    if args.benchmark in ["vhelm", "helm", "leaderboard"]:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
    elif args.benchmark == "lmms-eval":
        ax.set_ylim(0.0, 0.8)
        ax.set_yticks(np.arange(0.0, 0.81, 0.2))

    ax.set_xticks(np.arange(0, len(df["percentage"]), 2))
    ax.set_xlim(0, len(df["percentage"]) - 1)
    ax.set_xlabel("Fraction of Data Missing")
    if not is_subplot or args.benchmark == "helm":
        ax.set_ylabel(r"Kendall's $\tau$ coefficient")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    benchmark_to_title = {
        "vhelm": "VHELM",
        "helm": "HELM",
        "leaderboard": "Open LLM Leaderboard",
        "lmms-eval": "LMMs-Eval",
        "synthetic": "Synthetic",
    }
    ax.set_title(benchmark_to_title[args.benchmark], fontsize=15, pad=10)

    return fig, ax


def generate_markdown_table(df):
    """Generate a markdown table from the tau values."""
    # Round values to 2 decimal places
    df_rounded = df.with_columns(
        [
            pl.col("taus_mle").round(2),
            pl.col("taus_elo").round(2),
            pl.col("taus_lsr").round(2),
            pl.col("taus_mle_std").round(2),
            pl.col("taus_elo_std").round(2),
            pl.col("taus_lsr_std").round(2),
        ]
    )

    # Create header
    header = "| Fraction of Data Missing | ELO | LMArena | Ours |"
    separator = "|------------------------|-----|---------|------|"

    # Create rows
    rows = []
    for i, percentage in enumerate(df_rounded["percentage"]):
        row = (
            f"| {percentage:.2f} | "
            f"{df_rounded['taus_elo'][i]:.2f} ± {df_rounded['taus_elo_std'][i]:.2f} | "
            f"{df_rounded['taus_mle'][i]:.2f} ± {df_rounded['taus_mle_std'][i]:.2f} | "
            f"{df_rounded['taus_lsr'][i]:.2f} ± {df_rounded['taus_lsr_std'][i]:.2f} |"
        )
        rows.append(row)

    return "\n".join([header, separator] + rows)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=ROOT_DIR / "results/robustness/",
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
        default="vhelm",
        choices=["vhelm", "helm", "leaderboard", "lmms-eval", "synthetic", "all"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force recomputation of taus even if file exists",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        help="List of percentages for missing data (0-1). If not provided, uses default range.",
        default=None,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "svg"],
        default="pdf",
        help="Output format for figures (default: pdf)",
    )
    args = parser.parse_args()

    if args.benchmark == "all":
        plot_all_benchmarks(args)
        for benchmark in ["helm", "leaderboard", "vhelm", "lmms-eval"]:
            args.benchmark = benchmark
            df = load_or_compute_taus(args)
            print(f"\n### {benchmark.upper()}")
            print(generate_markdown_table(df))
    else:
        plot_single_benchmark(args)
        df = load_or_compute_taus(args)
        print("\n### Tau Values")
        print(generate_markdown_table(df))


if __name__ == "__main__":
    main()
