import os
import polars as pl
from elo import (
    compute_mle,
    to_pairwise,
    compute_elo,
    compute_plackett_luce_pairwise_dense,
)
import choix
import pandas as pd
from utils import *
from tqdm import tqdm
import random
from argparse import ArgumentParser

root_dir = ""


def results_path(dataset):
    if dataset == "vhelm" or dataset == "lmms-eval":
        return f"{root_dir}data/vlm/{dataset}/results.parquet"
    elif dataset == "helm" or dataset == "leaderboard":
        return f"{root_dir}data/llm/{dataset}/results.parquet"


def load_long(df):
    return df.melt(id_vars=["model"], variable_name="data_instance", value_name="score")


def compute_accuracy_ranking(results):
    """Rank the models by their average accuracy."""
    scaler = StandardScaler()
    results = results.to_pandas()
    results.set_index("model", inplace=True)
    numeric_columns = []
    for column in results.columns:
        unique_values = results[column].dropna().unique()
        if set(unique_values).issubset({0.0, 1.0}):
            continue
        else:
            numeric_columns.append(column)
    if numeric_columns:
        results[numeric_columns] = scaler.fit_transform(results[numeric_columns])

    model_scores = results.mean(axis=1)
    ranked_models = model_scores.rank(ascending=False, method="min")
    ranking = ranked_models.sort_values()
    return ranking


def compute_pairwise(pairwise_results):
    """Rank the models using the Plackett-Luce model based on pairwise results."""
    rankings = []

    models = (
        pl.concat([pairwise_results["model_a"], pairwise_results["model_b"]])
        .unique()
        .to_list()
    )
    model_to_index = {model: idx for idx, model in enumerate(models)}
    for model_a, model_b, winner in pairwise_results[
        ["model_a", "model_b", "winner"]
    ].iter_rows():
        if winner == "model_a":
            rankings.append([model_to_index[model_a], model_to_index[model_b]])
        elif winner == "model_b":
            rankings.append([model_to_index[model_b], model_to_index[model_a]])
        elif winner == "tie":
            rankings.append([model_to_index[model_b], model_to_index[model_a]])
            rankings.append([model_to_index[model_a], model_to_index[model_b]])
    return rankings, models


def calculate_precision(predicted_models, actual_models, N):
    # Get the top N from both lists
    top_n_predicted = set(predicted_models[:N])
    top_n_actual = set(actual_models[:N])

    # Calculate precision
    correct_predictions = top_n_predicted.intersection(top_n_actual)
    precision = len(correct_predictions) / N

    return precision


parser = ArgumentParser()
parser.add_argument(
    "--result_dir",
    type=str,
    help="Where to save the data",
    default="results/topk/",
)
parser.add_argument("--dataset", type=str, default="vhelm")
args = parser.parse_args()

os.makedirs(args.result_dir, exist_ok=True)
result_path = results_path(args.dataset)
results = pl.read_parquet(result_path)
ranking = rank_models(results)
ranking_list = list(ranking.index)
print(ranking_list)
with open(f"{args.result_dir}/{args.dataset}_gt.txt", "w") as file:
    file.write(str(ranking_list))

top10_elo = []
top10_mle = []
top10_lsr = []
percentages = [0.0, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
for fraction in tqdm(percentages):

    print("Fraction:", fraction)
    columns = results.columns
    sampled_columns = random.sample(columns, int((1 - fraction) * len(columns)))
    sampled = results.select(sampled_columns)
    if "model" not in sampled.columns:
        sampled = sampled.with_columns(results["model"].alias("model"))
    long_sampled = load_long(sampled)
    pairwise_sampled = to_pairwise(long_sampled, long=True)

    ranking_mle = compute_mle(pairwise_sampled.to_pandas())
    ranking_elo = compute_elo(pairwise_sampled.to_pandas())

    mle_list = list(ranking_mle.index)
    elo_list = list(ranking_elo.index)

    # with open(f"{args.result_dir}/{args.dataset}_ca_{fraction}.txt", 'w') as file:
    #     file.write(str(mle_list))
    # with open(f"{args.result_dir}/{args.dataset}_elo_{fraction}.txt", 'w') as file:
    #     file.write(str(elo_list))

    print("ELO and MLE done")
    if (
        args.dataset == "helm"
        or args.dataset == "leaderboard"
        or args.dataset == "lmms-eval"
    ):
        pair_list, models = compute_pairwise(pairwise_sampled)
        lsr_params = choix.lsr_pairwise(len(models), pair_list, alpha=0.05)
        lsr_rank = pd.Series(lsr_params, index=models).sort_values(ascending=False)
    else:
        sampled = sampled.to_pandas()
        sampled.set_index("model", inplace=True)
        top1_ranks = gen_top1_list(sampled)
        lsr_params = choix.lsr_top1(len(sampled.index), top1_ranks, alpha=0.05)
        lsr_rank = pd.Series(lsr_params, index=sampled.index).sort_values(
            ascending=False
        )
    print("finished LSR")

    lsr_list = list(lsr_rank.index)

    with open(f"{args.result_dir}/{args.dataset}_ca_{fraction}.txt", "w") as file:
        file.write(str(mle_list))
    with open(f"{args.result_dir}/{args.dataset}_elo_{fraction}.txt", "w") as file:
        file.write(str(elo_list))
    with open(f"{args.result_dir}/{args.dataset}_pl_{fraction}.txt", "w") as file:
        file.write(str(lsr_list))

    # Calculate top10 precision
    top10_elo.append(calculate_precision(elo_list, ranking_list, 10))
    top10_mle.append(calculate_precision(mle_list, ranking_list, 10))
    top10_lsr.append(calculate_precision(lsr_list, ranking_list, 10))

    print("LSR:", top10_lsr[-1])
    print("MLE:", top10_mle[-1])
    print("ELO:", top10_elo[-1])
    print()

# create dataframe with percenatges
df = pd.DataFrame(
    {
        "percentage": percentages,
        "top10_mle": top10_mle,
        "top10_elo": top10_elo,
        "top10_lsr": top10_lsr,
    }
)
# df.to_csv(f"{args.result_dir}/{args.dataset}_top10.csv")
