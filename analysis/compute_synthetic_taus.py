import argparse
import numpy as np
import polars as pl
from tqdm import tqdm

from analysis import (
    compute_elo,
    compute_mle,
    compute_plackett_luce_pairwise_dense,
    compute_tau,
    to_pairwise,
)
from analysis import generate_synthetic_data


def compute_taus_for_rankings(pairwise_data, global_ranking, n=5):
    """Compute the Kendall Tau correlation of the ranking with missing data."""
    ranking_mle = compute_mle(pairwise_data.to_pandas())
    ranking_elo = compute_elo(pairwise_data.to_pandas())
    ranking_lsr = compute_plackett_luce_pairwise_dense(pairwise_data)

    return {
        "mle": compute_tau(global_ranking, ranking_mle.index),
        "elo": compute_tau(global_ranking, ranking_elo.index),
        "lsr": compute_tau(global_ranking, ranking_lsr.index),
    }


def compute_and_print_taus(args):
    """Compute and print Kendall tau correlations for different ranking methods."""
    print("Computing Kendall tau correlations...")
    results = []

    for dispersion in tqdm(args.dispersions, desc="Processing different dispersions"):
        taus_all = []
        for _ in range(args.n):
            df, model_params = generate_synthetic_data(
                num_models=args.num_models,
                num_samples=args.num_samples,
                distribution=args.distribution,
                dispersion=dispersion,
            )
            pairwise_data = to_pairwise(pl.from_pandas(df), long=False)
            taus = compute_taus_for_rankings(pairwise_data, model_params["model"])
            taus_all.append(taus)

        avg_taus = {
            "mle": {
                "mean": np.mean([t["mle"] for t in taus_all]),
                "std": np.std([t["mle"] for t in taus_all]),
            },
            "elo": {
                "mean": np.mean([t["elo"] for t in taus_all]),
                "std": np.std([t["elo"] for t in taus_all]),
            },
            "lsr": {
                "mean": np.mean([t["lsr"] for t in taus_all]),
                "std": np.std([t["lsr"] for t in taus_all]),
            },
        }
        results.append((dispersion, avg_taus))

    print("\nKendall Tau Correlations:")
    print("| Dispersion | ELO | LMArena | Ours |")
    print("|-------------------|-----|---------|------|")
    for dispersion, taus in results:
        print(
            f"| {dispersion:.2f} | "
            f"{taus['elo']['mean']:.2f} ± {taus['elo']['std']:.2f} | "
            f"{taus['mle']['mean']:.2f} ± {taus['mle']['std']:.2f} | "
            f"{taus['lsr']['mean']:.2f} ± {taus['lsr']['std']:.2f} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute Kendall tau correlations for synthetic data"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of times to compute taus for averaging",
    )
    parser.add_argument(
        "--num_models", type=int, default=100, help="Number of models to generate"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--dispersions",
        type=float,
        nargs="+",
        default=[0.1],
        help="Dispersion values for model distributions",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="gaussian",
        choices=["gaussian", "gumbel"],
        help="Distribution type for model scores",
    )
    args = parser.parse_args()
    compute_and_print_taus(args)


if __name__ == "__main__":
    main()
