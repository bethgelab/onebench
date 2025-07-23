import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import polars as pl
from tqdm import tqdm

from elo import *

root_dir = '/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/'
def load_vlm_results(dataset):
    if dataset == 'vhelm':
        results_path = f'{root_dir}data/vlm/vhelm/results.parquet'
    else:
        results_path = f'{root_dir}data/vlm/lmms-eval/results.parquet'
    return pl.read_parquet(results_path)

def load_vlm_results_long(dataset):
    if dataset == 'vhelm':
        results_path = f'{root_dir}data/vlm/vhelm/results.parquet'
    else:
        results_path = f'{root_dir}data/vlm/lmms-eval/results.parquet'
    return pl.read_parquet(results_path).melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )

def load_vlm_results_pairwise(dataset):
    if dataset == 'vhelm':
        pairwise_results_path = f'{root_dir}data/vlm/vhelm/pairwise.parquet'
    else:
        pairwise_results_path = f'{root_dir}data/vlm/lmms-eval/pairwise.parquet'
    if os.path.exists(pairwise_results_path):
        pairwise_results = pl.read_parquet(pairwise_results_path)
    else:
        pairwise_results = to_pairwise(load_vlm_results_long(dataset))
        pairwise_results.write_parquet(pairwise_results_path)
    return pairwise_results

def main():
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default="results/rank_estimation/")
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument(
        "--benchmark", choices=["lmms-eval", "vhelm"], default="vhelm"
    )
    args = parser.parse_args()
    estimate_model_rank(args)


def estimate_model_rank(args):
    results = load_vlm_results(args.benchmark).to_pandas()
    # if args.benchmark == 'lmms-eval':
    results.set_index("model", inplace=True)
    pairwise_results = load_vlm_results_pairwise(args.benchmark)

    print(results)
    results["accuracy"] = results.mean(axis=1)
    results = results.sort_values(by="accuracy", ascending=False)
    results = results.drop(columns="accuracy")

    data_range, means_random, means_informative = load_or_compute_distances(
        args,
        pairwise_results,
        ranking=results.index,
        data_instances=results.columns,
        model_accuracy=results.mean(axis=0),
    )

    plt.style.use(["science", "vibrant", "grid"])
    plt.rcParams["font.family"] = "Times"
    fig, ax = plt.subplots()
    ax.plot(data_range, means_random, label="Random")
    ax.plot(data_range, means_informative, label="Informative")
    ax.set_xlabel("Number of data instances")
    ax.set_ylabel("Avg. distance to True Rank(VHELM)")
    ax.set_ylim(0, 5)
    ax.legend()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out_dir / f"rank_estimation_{args.benchmark}.pdf")


def load_or_compute_distances(
    args, pairwise_results, ranking, data_instances, model_accuracy
):
    """Compute the average distance of the ranking to the true ranking."""
    path = f"{root_dir}data/vlm/{args.benchmark}/model_rank_estimation.csv"

    if os.path.exists(path):
        df = pl.read_csv(path)
        return (
            df["data_range"].to_numpy(),
            df["means_random"].to_numpy(),
            df["means_informative"].to_numpy(),
        )

    means_random = []
    means_informative = []
    stds_random = []
    stds_informative = []
    data_range = list(range(10, 1001, 10))
    for num_data_instances in tqdm(data_range):
        random_data_instances = np.random.choice(data_instances, num_data_instances)
        informative_data_instances = select_middle_elements_by_accuracy(
            data_instances, model_accuracy, num_data_instances
        )
        random_ranking = compute_ranking_for_subset(
            pairwise_results, random_data_instances
        )
        informative_ranking = compute_ranking_for_subset(
            pairwise_results, informative_data_instances
        )
        dists_random = [
            abs(
                ranking.to_list().index(model)
                - random_ranking.index.to_list().index(model)
            )
            for model in ranking
        ]
        dists_informative = [
            abs(
                ranking.to_list().index(model)
                - informative_ranking.index.to_list().index(model)
            )
            for model in ranking
        ]
        means_random.append(np.mean(dists_random))
        means_informative.append(np.mean(dists_informative))
        stds_random.append(np.std(dists_random))
        stds_informative.append(np.std(dists_informative))

    df = pl.DataFrame(
        {
            "data_range": data_range,
            "means_random": means_random,
            "means_informative": means_informative,
            "stds_random": stds_random,
            "stds_informative": stds_informative,
        }
    )
    df.write_csv(path)

    return data_range, means_random, means_informative


def select_middle_elements_by_accuracy(data_instances, accuracies, m):
    """Select m data instances with model accuracy closest to 0.5."""
    data_instances = np.array(data_instances)
    accuracies = np.array(accuracies)

    sorted_indices = np.argsort(np.abs(accuracies - 0.5))
    sorted_data_instances = data_instances[sorted_indices]

    return np.random.choice(sorted_data_instances[: 10 * m], m, replace=False)


if __name__ == "__main__":
    main()
