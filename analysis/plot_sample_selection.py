"""Compare the performance of random and informative sampling."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import polars as pl
from tqdm import tqdm

from analysis import (
    compute_ranking_for_subset_dense,
    compute_score_ranking,
    compute_tau,
)
from analysis import load_results, load_pairwise_results


def main():
    root = Path(__file__).parents[2]
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=root / "results")
    parser.add_argument("--fig_path", type=Path, default=root / "figures")
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument(
        "--benchmark",
        choices=["helm", "leaderboard", "vhelm", "lmms-eval", "all"],
        default="all",
    )
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    args = parser.parse_args()
    plot_sample_selection(args)


def plot_sample_selection(args):
    """Plot the performance of random and informative sampling."""
    if args.benchmark == "all":
        plot_all_benchmarks(args)
    else:
        plot_single_benchmark(args)


def plot_all_benchmarks(args):
    """Plot all benchmarks as subplots in a single figure with a shared legend."""
    benchmarks = ["helm", "leaderboard", "vhelm", "lmms-eval"]
    plt.style.use(["science", "vibrant", "grid"])
    plt.rcParams["font.family"] = "Times"
    fig, axes = plt.subplots(
        1, 4, figsize=(args.fig_width * len(benchmarks), args.fig_height * 1.2)
    )

    for ax, benchmark in zip(axes, benchmarks):
        args.benchmark = benchmark
        plot_single_benchmark(args, ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    handles = [
        plt.Line2D([], [], color=handle.get_color(), linewidth=2.0)
        for handle in handles
    ]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=14,
    )

    fig.tight_layout()
    path = args.fig_path / "sample_selection.pdf"
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight")


def plot_single_benchmark(args, ax=None):
    """Plot the performance of random and informative sampling for a single benchmark."""
    df = load_or_compute_taus(args)
    plt.style.use(["science", "vibrant", "grid"])
    plt.rcParams["font.family"] = "Times"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.plot(df["data_range"], df["means_random"], label="Random")
    ax.fill_between(
        df["data_range"],
        df["means_random"] - df["stds_random"],
        df["means_random"] + df["stds_random"],
        alpha=0.2,
    )
    ax.plot(df["data_range"], df["means_informative"], label="Informative")
    ax.fill_between(
        df["data_range"],
        df["means_informative"] - df["stds_informative"],
        df["means_informative"] + df["stds_informative"],
        alpha=0.2,
    )
    ax.set_xlabel("Number of data samples")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    if fig is not None or args.benchmark == "helm":
        ax.set_ylabel(r"Kendall $\tau$ coefficient")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    benchmark_to_title = {
        "vhelm": "VHELM",
        "helm": "HELM",
        "leaderboard": "Open LLM Leaderboard",
        "lmms-eval": "LMMs-Eval",
    }
    ax.set_title(benchmark_to_title[args.benchmark], fontsize=15, pad=10)

    if fig is not None:
        path = args.fig_path / f"sample_selection_{args.benchmark}.pdf"
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(path)


def load_or_compute_taus(args):
    """Compute the average distance of the ranking to the true ranking."""
    path = args.data_path / f"sample_selection/{args.benchmark}/kendall_taus.csv"
    if path.exists():
        return pl.read_csv(path)

    results = load_results(args.benchmark)
    pairwise_results = load_pairwise_results(args.benchmark)
    data_samples = results.drop("model").columns
    global_ranking = compute_score_ranking(results, args.benchmark)
    percentage_correct = results.drop("model").mean().to_numpy().flatten() * 100

    means_random = []
    means_informative = []
    stds_random = []
    stds_informative = []
    data_range = [10] + list(range(100, 10001, 100))

    for num_samples in tqdm(data_range):

        taus_random = []
        taus_informative = []

        for _ in range(5):
            random_samples = np.random.choice(data_samples, num_samples)
            informative_samples = select_informative_samples(
                data_samples, percentage_correct, num_samples
            )
            try:
                random_ranking = compute_ranking_for_subset_dense(
                    pairwise_results, random_samples
                )
                informative_ranking = compute_ranking_for_subset_dense(
                    pairwise_results, informative_samples
                )
                taus_random.append(
                    compute_tau(global_ranking.index, random_ranking.index)
                )
                taus_informative.append(
                    compute_tau(global_ranking.index, informative_ranking.index)
                )
            except (ValueError, RuntimeError, RuntimeWarning):
                continue

        means_random.append(np.mean(taus_random))
        means_informative.append(np.mean(taus_informative))
        stds_random.append(np.std(taus_random))
        stds_informative.append(np.std(taus_informative))

    df = pl.DataFrame(
        {
            "data_range": data_range,
            "means_random": means_random,
            "means_informative": means_informative,
            "stds_random": stds_random,
            "stds_informative": stds_informative,
        }
    )
    path.parent.mkdir(exist_ok=True, parents=True)
    df.write_csv(path)

    return df


def select_informative_samples(data_samples, percentage_correct, m):
    """Select m data samples that are closest to 50% of models solving them."""
    data_samples = np.array(data_samples)
    percentage_correct = np.array(percentage_correct)

    sorted_indices = np.argsort(np.abs(percentage_correct - 50))
    sorted_data_samples = data_samples[sorted_indices]

    return np.random.choice(sorted_data_samples[: 10 * m], m, replace=False)


if __name__ == "__main__":
    main()
