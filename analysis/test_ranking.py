import argparse
from analysis import load_results, load_pairwise_results
from analysis import (
    compute_plackett_luce_pairwise_dense,
    compute_score_ranking_dense,
    compute_tau,
)
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).parents[2] / "data/llm"


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_n", type=int, default=5)
    args = parser.parse_args()
    test_ranking(args)


def test_ranking(args):
    helm_to_leaderboard = {
        "01-ai_yi-34b": "01-ai/yi-34b-200k",
        "google_gemma-7b": "google/gemma-7b",
        "meta_llama-3-70b": "meta-llama/meta-llama-3-70b",
        "meta_llama-3-8b": "meta-llama/meta-llama-3-8b",
        "microsoft_phi-2": "microsoft/phi-2",
        "mistralai_mixtral-8x22b": "mistral-community/mixtral-8x22b-v0.1",
        "qwen_qwen1.5-14b": "qwen/qwen1.5-14b",
        "qwen_qwen1.5-7b": "qwen/qwen1.5-7b",
    }

    results_helm = load_results(benchmark="helm")
    pairwise_results_helm = load_pairwise_results(benchmark="helm")

    results_leaderboard = load_results(benchmark="leaderboard")
    pairwise_results_leaderboard = load_pairwise_results(benchmark="leaderboard")

    results_helm = results_helm.with_columns(
        pl.col("model").replace(helm_to_leaderboard).alias("model")
    )
    pairwise_results_helm = pairwise_results_helm.with_columns(
        pl.col("model_a").replace(helm_to_leaderboard).alias("model_a"),
        pl.col("model_b").replace(helm_to_leaderboard).alias("model_b"),
    )

    print("Computing HELM ranking...")
    helm_ranking = compute_plackett_luce_pairwise_dense(pairwise_results_helm)

    print("Computing HELM accuracy ranking...")
    helm_ranking_acc = compute_score_ranking_dense(results_helm, long=False)

    tau = compute_tau(helm_ranking.index, helm_ranking_acc.index)

    print(helm_ranking[: args.show_n], "\n")
    print(helm_ranking_acc[: args.show_n], "\n")
    print(f"Tau: {tau}\n")

    print("Computing Leaderboard ranking...")
    leaderboard_ranking = compute_plackett_luce_pairwise_dense(
        pairwise_results_leaderboard
    )
    print("Computing Leaderboard accuracy ranking...")
    leaderboard_ranking_acc = compute_score_ranking_dense(
        results_leaderboard, long=False
    )
    tau = compute_tau(leaderboard_ranking.index, leaderboard_ranking_acc.index)

    print(leaderboard_ranking[: args.show_n], "\n")
    print(leaderboard_ranking_acc[: args.show_n], "\n")
    print(f"Tau: {tau}", "\n")

    concat_results = pl.concat([pairwise_results_helm, pairwise_results_leaderboard])

    print("Computing joint ranking...")
    results_pairwise = load_pairwise_results()
    ranking_dense_raw = compute_plackett_luce_pairwise_dense(results_pairwise)
    print(ranking_dense_raw[: args.show_n])
    print()

    print("Computing joint ranking from concat...")
    ranking_dense = compute_plackett_luce_pairwise_dense(concat_results)
    print(ranking_dense[: args.show_n])


if __name__ == "__main__":
    _main()
