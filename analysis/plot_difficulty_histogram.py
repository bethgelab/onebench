from argparse import ArgumentParser
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import polars as pl

from analysis import (
    compute_ranking_for_subset_dense,
    compute_tau,
    compute_score_ranking,
)
from analysis import load_results, load_pairwise_results

plt.style.use(["science", "vibrant", "grid"])
plt.rcParams["font.family"] = "Times"


def main():
    root = Path(__file__).parents[2]
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=root / "results")
    parser.add_argument("--fig_path", type=Path, default=root / "figures")
    parser.add_argument("--num_bins", type=int, nargs="+", default=30)
    parser.add_argument(
        "--benchmark",
        choices=["helm", "leaderboard", "vhelm", "lmms-eval", "all"],
        default="all",
    )
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    args = parser.parse_args()

    if args.benchmark == "all":
        plot_all_benchmarks(args)
    else:
        plot_single_benchmark(args)


def plot_all_benchmarks(args):
    """Plot histograms for multiple benchmarks as subplots in a single figure."""
    benchmarks = ["helm", "leaderboard", "vhelm", "lmms-eval"]
    num_plots = len(benchmarks)
    fig, axes = plt.subplots(
        1,
        num_plots,
        figsize=(args.fig_width * num_plots, args.fig_height * 1.2),
    )
    if isinstance(args.num_bins, int):
        args.num_bins = [args.num_bins] * num_plots

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax1, benchmark, num_bins in zip(axes, benchmarks, args.num_bins):
        args.benchmark = benchmark
        args.num_bins = num_bins
        ax2 = ax1.twinx()
        plot_single_benchmark(args, ax1, ax2, cycle)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        ncol=len(benchmarks),
        bbox_to_anchor=(0.5, -0.15),
        fontsize=14,
    )
    fig.tight_layout()
    path = args.fig_path / "difficulty_histograms.pdf"
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight")


def plot_single_benchmark(args, ax1=None, ax2=None, cycle=None):
    """Plot the difficulty histogram for a single benchmark."""
    percentage_correct_path = (
        args.data_path / f"histogram/{args.benchmark}/percentage_correct.npy"
    )
    if percentage_correct_path.exists():
        print("Loading existing percentage_correct data")
        percentage_correct = np.load(percentage_correct_path)
    else:
        results = load_results(benchmark=args.benchmark)
        if args.benchmark == "lmms-eval":
            threshold = 0.3 #good threshold for rouge
            # Normalize 0-10 scale columns to 0-1
            results_normalized = results.with_columns(
                [
                    pl.when(pl.col(col).max() >= 2)  # Check if the column has a max of 10
                    .then(pl.col(col) / 10)
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in results.columns if col != "model"
                ]
            )

            # convert all columns to binary correctness based on the threshold
            results_binary = results_normalized.with_columns(
                [
                    pl.when(pl.col(col) >= threshold).then(1).otherwise(0).alias(col)
                    for col in results_normalized.columns if col != "model"
                ]
            )

            # calculate the percentage of models correct for each data instance
            percentage_correct = (
                    results_binary.select([pl.exclude("model")])
                    .select(
                        [pl.col(col).mean().alias(col) for col in results_binary.columns if col != "model"]
                    )
                    .to_numpy() 
                    * 100
            ).flatten()
        else:
            percentage_correct = results.drop("model").mean().to_numpy().flatten() * 100
        np.save(percentage_correct_path, percentage_correct)

    counts, bins = np.histogram(percentage_correct, bins=args.num_bins, range=(0, 100))
    df = compute_taus(args, percentage_correct, bins)

    if ax1 is None or ax2 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(args.fig_width, args.fig_height))
        ax2 = ax1.twinx()
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        is_subplot = False
    else:
        is_subplot = True

    # plot the histogram
    ax1.hist(
        percentage_correct,
        bins=args.num_bins,
        range=(0, 100),
        color=cycle[0],
        label="Number of Data Samples",
    )

    right_ticks = np.arange(0, 1.1, 0.2)
    left_ticks = right_ticks * (np.ceil(max(counts) / 1000) * 1000)

    ax1.set_yticks(left_ticks)
    ax1.set_ylim(min(left_ticks), max(left_ticks))
    ax1.yaxis.set_label_coords(-0.1, 0.5)

    ax1.set_xlabel("Percentage of Models Correctly Predicting")
    ax1.set_xticks(np.arange(0, 110, 20))
    ax1.grid(True)

    benchmark_to_title = {
        "vhelm": "VHELM",
        "helm": "HELM",
        "leaderboard": "Open LLM Leaderboard",
        "lmms-eval": "LMMs-Eval",
    }
    ax1.set_title(benchmark_to_title[args.benchmark], fontsize=15, pad=10)
    if not is_subplot or args.benchmark == "helm":
        ax1.set_ylabel("Percentage of Data Samples")
        ax1.yaxis.set_label_position("left")
        ax1.yaxis.set_label_coords(-0.13, 0.5)

    # plot the taus
    ax2.plot(
        df["accs"],
        df["taus"],
        marker="o",
        markersize=3,
        linestyle="-",
        color=cycle[1],
        label=r"Kendall's $\tau$ coefficient",
    )
    ax2.set_yticks(right_ticks)
    ax2.set_ylim(min(right_ticks), max(right_ticks))
    ax2.grid(False)

    if not is_subplot or args.benchmark == "lmms-eval":
        ax2.set_ylabel(r"Kendall's $\tau$ coefficient")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_label_coords(1.13, 0.5)

    if not is_subplot:
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(
            args.fig_path / f"difficulty_histograms{args.benchmark}.pdf",
            bbox_inches="tight",
        )


def sort_by_model_accuracy(results):
    """Sort the models by accuracy."""
    return (
        results.with_columns(pl.mean_horizontal(pl.exclude("model")).alias("accuracy"))
        .sort("accuracy", descending=True)
        .drop("accuracy")
    )


def compute_taus(args, percentage_correct, bins):
    """Compute ranking and its correlation to the global ranking for each bin."""
    path = args.data_path / f"histogram/{args.benchmark}/kendall_taus.csv"
    if path.exists():
        print("Loading existing taus data")
        df = pl.read_csv(path)
        assert len(df["taus"]) == len(df["accs"]) == args.num_bins, (
            "Number of taus and accs must match number of bins. "
            f"Got {len(df['taus'])} and {args.num_bins}"
        )
        return df.drop_nulls()

    results = load_results(benchmark=args.benchmark)
    global_ranking = compute_score_ranking(results, args.benchmark)
    path = args.data_path / f"histogram/{args.benchmark}/global_ranking.csv"
    global_ranking.to_csv(path, header=True, index=True)

    results = load_results(benchmark=args.benchmark)
    pairwise_results = load_pairwise_results(benchmark=args.benchmark)
    counts, _ = np.histogram(percentage_correct, bins=args.num_bins, range=(0, 100))

    taus = []
    accs = []
    for count, start, end in zip(counts, bins[:-1], bins[1:]):
        indices = get_bin_indices(percentage_correct, start, end)
        assert (
            len(indices) == count
        ), f"Number of indices is {len(indices)} but should be {count}"

        middle = (start + end) / 2
        try:
            ranking = compute_ranking_for_subset_dense(
                pairwise_results, results.to_pandas().columns[indices].to_list()
            )
            tau = compute_tau(ranking.index, global_ranking.index)
        except (np.linalg.LinAlgError, ValueError, IndexError, RuntimeError) as e:
            print(
                f"Error computing tau for {args.benchmark} with start: {start}, end: {end}: {e}"
            )
            tau = np.nan
        taus.append(tau)
        accs.append(middle)
    df = pl.DataFrame({"accs": accs, "taus": taus})
    path = args.data_path / f"histogram/{args.benchmark}/kendall_taus.csv"
    path.parent.mkdir(exist_ok=True, parents=True)
    df.write_csv(path)
    return df


def get_bin_indices(array, start, end):
    """Get the indices of the array that are in the bin."""
    indices = np.where((array >= start) & (array < end))[0]
    if np.isclose(end, np.max(array), rtol=1e-9, atol=1e-9):
        indices = np.where((array >= start) & (array <= end))[0]
    return indices


def darken_color(hex_color, scale=0.8):
    """Darken a color by a given scale."""
    rgb_color = mcolors.hex2color(hex_color)
    rgb_darken = tuple(max(0, min(1, c * scale)) for c in rgb_color)
    return mcolors.rgb2hex(rgb_darken)


if __name__ == "__main__":
    main()
