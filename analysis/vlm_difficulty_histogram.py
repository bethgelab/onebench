import json
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

plt.rc("text", usetex=False)

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import scienceplots

mpl.rcParams.update(mpl.rcParamsDefault)

from elo import *
from vlm_rank_change import *

root_dir = "/mnt/qb/work/bethge/bkr536/lifelong/"

plt.style.use(["science", "vibrant", "grid"])
plt.rcParams["font.family"] = "Times"


def load_or_compute_global_ranking():
    """Load or compute the global Bradley-Terry ranking."""
    ranking_path = VHELM_DIR + "ranking.csv"
    if os.path.exists(ranking_path):
        ranking = pd.read_csv(ranking_path, index_col=0)
    else:
        pairwise = load_vhelm_results_pairwise()
        ranking = compute_mle(pairwise)
        ranking.to_frame().to_csv(ranking_path)
    return ranking


def load_vlm_results(dataset):
    if dataset == "vhelm":
        results_path = f"{root_dir}data/vlm/vhelm/results.parquet"
    else:
        results_path = f"{root_dir}data/vlm/lmms-eval/results.parquet"
    return pl.read_parquet(results_path)


def load_vlm_results_long(dataset):
    if dataset == "vhelm":
        results_path = f"{root_dir}data/vlm/vhelm/results.parquet"
    else:
        results_path = f"{root_dir}data/vlm/lmms-eval/results.parquet"
    return pl.read_parquet(results_path).melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )


def load_vlm_results_pairwise(dataset):
    if dataset == "vhelm":
        pairwise_results_path = f"{root_dir}data/vlm/vhelm/pairwise.parquet"
    else:
        pairwise_results_path = f"{root_dir}data/vlm/lmms-eval/pairwise.parquet"
    if os.path.exists(pairwise_results_path):
        pairwise_results = pl.read_parquet(pairwise_results_path)
    else:
        pairwise_results = to_pairwise(load_vlm_results_long(dataset))
        pairwise_results.write_parquet(pairwise_results_path)
    return pairwise_results


def main():
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results/two_towers/")
    parser.add_argument("--num_data_instances", type=int, default=5)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="lmms-eval")
    args = parser.parse_args()
    print(os.getcwd())
    plot_difficulty_histogram(args)


def plot_difficulty_histogram(args):
    dataset = args.dataset

    # Check if file exists
    if not os.path.exists(args.out_dir + f"{dataset}_accuracy_histogram.json"):
        print("Loading the results parquet")

        data_path = f"data/vlm/{dataset}/results.parquet"

        data = load_vlm_results(dataset).to_pandas()
        pairwise_results = load_vlm_results_pairwise(dataset)

        data.set_index("model", inplace=True)
        data["accuracy"] = data.mean(axis=1)
        data = data.sort_values(by="accuracy", ascending=False)
        data = data.drop(columns="accuracy")
        print(data)
        os.makedirs(args.out_dir, exist_ok=True)
        # global_ranking = load_or_compute_global_ranking()
        # global_ranking = data

        model_accuracy = data.mean(axis=0).to_numpy()
        print("got model accuracy")

        _, bins = np.histogram(model_accuracy, bins=args.num_bins)

        taus = []
        accs = []
        for start, end in zip(bins[:-1], bins[1:]):
            bin_indices = np.where((model_accuracy >= start) & (model_accuracy < end))[
                0
            ]
            ranking = compute_ranking_for_subset(
                pairwise_results, data.columns[bin_indices].to_list()
            )
            # print(ranking.index)
            # print(data.index)
            # ranking = compute_plackett_luce(data[data.columns[bin_indices].to_list()].reset_index())

            tau = compute_tau(data.index, ranking.index)
            # tau = compute_tau(global_ranking.index, ranking.index)
            middle = (start + end) / 2
            taus.append(tau)
            accs.append(middle)
        print("got taus and accs")

        # save model_accuracy, taus and accs to a csv file
        acc_dict = {"model_accuracy": list(model_accuracy), "taus": taus, "accs": accs}
        with open(
            f"{args.out_dir}/{dataset}_{args.num_bins}_accuracy_histogram.json", "w"
        ) as outfile:
            json.dump(acc_dict, outfile, indent=4)
    else:
        print("Loading the results from json")
        filename = f"{args.out_dir}/{dataset}_accuracy_histogram.json"
        with open(filename, "r") as infile:
            data = json.load(infile)

        model_accuracy = data["model_accuracy"]
        taus = list(data["taus"])
        accs = list(data["accs"])

    right_ticks = np.linspace(0, 1, 11)
    if args.dataset == "lmms-eval":
        limit = 150000
    else:
        limit = 4000
    left_ticks = right_ticks * limit

    # plot the histogram
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    width = 0.8 * (np.max(model_accuracy) - np.min(model_accuracy)) / args.num_bins
    print(cycle[0])
    ax1.hist(
        model_accuracy,
        bins=args.num_bins,
        color=cycle[0],
        edgecolor=darken_color(cycle[0], 1.0),
        width=width,
    )
    ax1.set_ylabel("Number of Data Instances")
    ax1.set_xlabel("Percentage of Models Correctly Predicting")
    ax1.set_ylim(0, limit)
    ax1.set_yticks(left_ticks)
    ax1.grid(True)

    # plot the taus
    ax2 = ax1.twinx()
    ax2.plot(accs, taus, marker="o", linestyle="-", color=cycle[1])
    ax2.set_ylabel("Kendall rank correlation coefficient")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(right_ticks)
    ax2.grid(False)
    fig.savefig(args.out_dir + f"{dataset}_accuracy_histogram.pdf")


def darken_color(hex_color, scale=0.8):
    """Darken a color by a given scale."""
    rgb_color = mcolors.hex2color(hex_color)
    rgb_darken = tuple(max(0, min(1, c * scale)) for c in rgb_color)
    return mcolors.rgb2hex(rgb_darken)


if __name__ == "__main__":
    main()
