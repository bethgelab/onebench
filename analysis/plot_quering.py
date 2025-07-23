from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from query import query
from analysis import (
    compute_ranking_for_subset_dense,
    compute_tau,
)
from analysis import (
    load_pairwise_results,
    load_or_compute_global_ranking,
)

ROOT = Path(__file__).parents[2]


def plot_querying(args):
    if args.queries_file:
        with open(args.queries_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries_colors = {
            "particle": "tab:blue",
            "topology": "tab:orange",
            "culture": "tab:green",
        }
        queries = list(queries_colors.keys())

    pairwise_results = load_pairwise_results(benchmark=args.benchmark)
    uuids = pairwise_results["data_instance"].unique().to_list()
    print("Loading global ranking...")
    global_ranking = load_or_compute_global_ranking(benchmark=args.benchmark)
    print("Done.")

    taus_random_means = []
    taus_random_stds = []
    taus_query = defaultdict(list)
    avg_similarities = defaultdict(list)
    sizes = args.sizes

    for k in tqdm(sizes, desc="Processing sizes"):
        taus_random = []
        for _ in tqdm(range(5), desc=f"Random sampling for k={k}", leave=False):
            uuids_random = np.random.choice(uuids, size=k, replace=False)
            ranking_random = compute_ranking_for_subset_dense(
                pairwise_results, uuids_random
            )
            if args.top_n_models is not None:
                ranking_random = ranking_random[: args.top_n_models]
                global_ranking_subset = global_ranking[: args.top_n_models]
            else:
                global_ranking_subset = global_ranking
            taus_random.append(
                compute_tau(ranking_random.index, global_ranking_subset.index)
            )
        taus_random_means.append(np.mean(taus_random))
        taus_random_stds.append(np.std(taus_random))

        for q in tqdm(queries, desc=f"Processing queries for k={k}", leave=False):
            config = DictConfig(
                {
                    "query": q,
                    "top_k": k,
                    "threshold": 0.0,
                    "filters": {},
                    "return_similarity": True,
                    "answer_query": None,
                }
            )
            config = OmegaConf.create(config)
            similarities, query_uuids = query(config)
            ranking = compute_ranking_for_subset_dense(pairwise_results, query_uuids)
            ranking = list(ranking.index)
            global_ranking_list = list(global_ranking.index)
            common_models = list(set(ranking) & set(global_ranking_list))
            ranking = [model for model in ranking if model in common_models]
            global_ranking_list = [
                model for model in global_ranking_list if model in common_models
            ]
            if args.top_n_models is not None:
                ranking = ranking[: args.top_n_models]
                global_ranking_list = global_ranking_list[: args.top_n_models]
            taus_query[q].append(compute_tau(ranking, global_ranking_list))
            avg_similarities[q].append(np.mean(similarities[similarities != -1]))

    plt.style.use(["science", "vibrant", "grid"])
    plt.rcParams["font.family"] = "Times"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(args.fig_width, args.fig_height * 2))

    ax1.plot(sizes, taus_random_means, label="random")
    ax1.fill_between(
        sizes,
        np.array(taus_random_means) - np.array(taus_random_stds),
        np.array(taus_random_means) + np.array(taus_random_stds),
        alpha=0.3,
        color="tab:red",
    )

    if args.queries_file:
        taus_avg = np.nanmean([taus_query[q] for q in queries], axis=0)
        taus_std = np.nanstd([taus_query[q] for q in queries], axis=0)
        similarities_avg = np.nanmean([avg_similarities[q] for q in queries], axis=0)
        similarities_std = np.nanstd([avg_similarities[q] for q in queries], axis=0)
        ax1.plot(sizes, taus_avg, label="average", color="tab:blue")
        ax1.fill_between(
            sizes,
            np.array(taus_avg) - np.array(taus_std),
            np.array(taus_avg) + np.array(taus_std),
            alpha=0.3,
            color="tab:blue",
        )
        ax2.plot(sizes, similarities_avg, label="average", color="tab:blue")
        ax2.fill_between(
            sizes,
            np.array(similarities_avg) - np.array(similarities_std),
            np.array(similarities_avg) + np.array(similarities_std),
            alpha=0.3,
            color="tab:blue",
        )
    else:
        for q, color in queries_colors.items():
            ax1.plot(sizes, taus_query[q], label=q, color=color)
            ax2.plot(sizes, avg_similarities[q], label=q, color=color)

    ax1.set_xscale("log")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(sizes)
    ax1.set_xlabel("Number of instances retrieved")
    ax1.set_ylabel(r"Kendall rank correlation $\tau$ to the full ranking")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xticks(sizes)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel("Number of instances retrieved")
    ax2.set_ylabel("Average similarity")
    ax2.legend()

    fig.savefig(ROOT / "figures/querying_with_similarity.pdf", bbox_inches="tight")

    print("\nSummary of results:")
    print("Size | Random tau | Query tau")
    print("-" * 30)
    for i, k in enumerate(sizes):
        if args.queries_file:
            taus_avg = np.nanmean([taus_query[q][i] for q in queries])
            taus_std = np.nanstd([taus_query[q][i] for q in queries])
            print(
                f"{k:5d} | {taus_random_means[i]:.3f} ± {taus_random_stds[i]:.3f} | {taus_avg:.3f} ± {taus_std:.3f}"
            )
        else:
            print(f"\nSize {k}:")
            print(
                f"Random sampling tau: {taus_random_means[i]:.3f} ± {taus_random_stds[i]:.3f}"
            )
            for q in queries:
                print(f"{q} tau: {taus_query[q][i]:.3f}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--fig_width", type=float, default=8)
    parser.add_argument("--fig_height", type=float, default=6)
    parser.add_argument("--benchmark", type=str, default="all")
    parser.add_argument(
        "--queries_file",
        type=str,
        help="Path to file containing queries (one per line)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument(
        "--top_n_models",
        type=int,
        default=None,
        help="If specified, only compute Kendall tau correlation for the top N models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_querying(args)
