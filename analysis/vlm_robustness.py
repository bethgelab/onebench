from argparse import ArgumentParser
import polars as pl
import scienceplots  # noqa: F401
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from elo import compute_mle, to_pairwise, compute_elo, compute_tau, compute_plackett_luce_robust
from visualization import visualize_pairwise_comparisons

VHELM_DIR = "data/vlm/vhelm/"
LMMS_EVAL_DIR = "data/vlm/lmms-eval/"
ARENA_DIR = "data/vlm/"

root_dir = '/mnt/qb/work/bethge/bkr536/lifelong/'


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


def compute_ranking_robustness(results_long):
    """Compute the Kendall Tau correlation of the ranking with missing data."""
    ranking = compute_accuracy_ranking(results_long)

    taus_mle_mu = []
    taus_elo_mu = []
    taus_pl_mu = []

    taus_mle_std = []
    taus_elo_std = []
    taus_pl_std = []

    percentages = list(np.linspace(0, 0.9, 10))


    for fraction in tqdm(percentages):
        print("Fraction:", fraction)
        taus_mle = []
        taus_elo = []
        taus_pl = []
        for _ in range(3):
            long_sampled = results_long.sample(fraction=1 - fraction, shuffle=True)
            pairwise_sampled = to_pairwise(long_sampled)

            ranking_mle = compute_mle(pairwise_sampled.to_pandas())
            ranking_elo = compute_elo(pairwise_sampled.to_pandas())
            ranking_pl = compute_plackett_luce_robust(pairwise_sampled)

            tau_mle = compute_tau(ranking["model"], ranking_mle.index)
            tau_elo = compute_tau(ranking["model"], ranking_elo.index)
            tau_pl = compute_tau(ranking["model"], ranking_pl.index)

            taus_mle.append(tau_mle)
            taus_elo.append(tau_elo)
            taus_pl.append(tau_pl)
        taus_mle_mu.append(np.mean(taus_mle))
        taus_elo_mu.append(np.mean(taus_elo))
        taus_pl_mu.append(np.mean(taus_pl))

        taus_mle_std.append(np.std(taus_mle))
        taus_elo_std.append(np.std(taus_elo))
        taus_pl_std.append(np.std(taus_pl))

        print("Bradley-Terry:", taus_mle_mu[-1], taus_mle_std[-1])
        print("ELO:", taus_elo_mu[-1], taus_elo_std[-1])
        print("Plackett-Luce:", taus_pl_mu[-1], taus_pl_std[-1])
        print()

    return pl.DataFrame(
        {
            "percentage": percentages,
            "taus_mle": taus_mle_mu,
            "taus_elo": taus_elo_mu,
            "taus_pl": taus_pl_mu,
            "taus_mle_std": taus_mle_std,
            "taus_elo_std": taus_elo_std,
            "taus_pl_std": taus_pl_std,
        }
    )


def drop_random(df, percentage, models):
    """Drop a random percentage of the data."""
    sampled_models = set()
    while not sampled_models == models:
        fraction = (100 - percentage) / 100
        df_sampled = df.sample(fraction=fraction, shuffle=True, seed=42)
        if "model" in df.columns:
            sampled_models = set(df_sampled["model"])
        elif "model_a" in df.columns and "model_b" in df.columns:
            sampled_models = set(df_sampled["model_a"]) | set(df_sampled["model_b"])
    return df_sampled


def compute_accuracy_ranking(df):
    """Rank the models by their average accuracy."""
    return (
        df.group_by("model")
        .agg(pl.mean("score").alias("mean_score"))
        .sort("mean_score", descending=True)
    )


def visualize(pairwise):
    """Visualize the pairwise score counts."""
    pairwise_pd = pairwise.to_pandas()
    visualize_pairwise_comparisons(pairwise_pd, "Model Battles").show()
    visualize_pairwise_comparisons(
        pairwise[pairwise_pd["winner"].str.contains("tie")],
        "Tie Count for Each Combination of Models",
    ).show()


def plot_robustness(args, data):
    """Plot the robustness of the ranking to missing data."""
    plt.style.use(["science", "vibrant", "grid"])
    plt.rcParams["font.family"] = "Times"
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.plot(
        data["percentage"],
        data["taus_mle"],
        label="Bradley-Terry Ranking",
    )
    ax.plot(
        data["percentage"],
        data["taus_elo"],
        label="ELO Ranking",
    )
    ax.plot(
        data["percentage"],
        data["taus_pl"],
        label="Plackett-Luce Ranking",
    )

    ax.fill_between(
        data["percentage"],
        np.array(data["taus_mle"]) - np.array(data["taus_mle_std"]),
        np.array(data["taus_mle"]) + np.array(data["taus_mle_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        data["percentage"],
        np.array(data["taus_elo"]) - np.array(data["taus_elo_std"]),
        np.array(data["taus_elo"]) + np.array(data["taus_elo_std"]),
        alpha=0.5,
    )
    ax.fill_between(
        data["percentage"],
        np.array(data["taus_pl"]) - np.array(data["taus_pl_std"]),
        np.array(data["taus_pl"]) + np.array(data["taus_pl_std"]),
        alpha=0.5,
    )
    ax.set_xlabel("Fraction of Data Missing")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.suptitle("Kendall Tau Correlation with Full Ranking (VHELM)")
    return fig


def test_ranking_robustness(args):
    """Test the robustness of the ranking to missing data."""

    # results_long = load_or_compute_kendall_taus(args, results_long, results_pairwise)
    if os.path.exists(root_dir + args.out_dir + f"{args.dataset}/kendall_taus.csv"):
        taus = pl.read_csv(root_dir + args.out_dir + f"{args.dataset}/kendall_taus.csv")

    else:
        results_long = load_vlm_results_long(args.dataset)
        print(results_long)
        # results_pairwise = load_vlm_results_pairwise(args.dataset)

        # if args.visualize:
        #     visualize(results_pairwise)
        taus = compute_ranking_robustness(results_long)
        taus.write_csv(root_dir + args.out_dir + f"{args.dataset}/kendall_taus.csv")

    fig = plot_robustness(args, taus)
    fig.savefig(root_dir + args.out_dir + f"{args.dataset}_kendall_taus.pdf", bbox_inches="tight")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Where to save the data",
        default="results/robust/",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--fig_width", type=float, default=4)
    parser.add_argument("--fig_height", type=float, default=3)
    parser.add_argument("--dataset", type=str, default="vhelm")
    args = parser.parse_args()

    test_ranking_robustness(args)


if __name__ == "__main__":
    main()

