import math
from collections import defaultdict

import choix
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def to_pairwise(results, long=True):
    """Convert cardinal data in long format to pairwise data."""
    if not long:
        results = results.melt(
            id_vars=["model"], variable_name="data_instance", value_name="score"
        )
    results = results.drop_nulls()

    data_instances = results["data_instance"].unique().to_list()

    pairwise = []
    for data_instance in tqdm(data_instances):
        instance_df = results.filter(pl.col("data_instance") == data_instance)

        # Create a self join to get pairs of models for the same data instance
        instance_df = instance_df.join(instance_df, on="data_instance", suffix="_other")

        # Filter out self-pairs and duplicate pairs
        filtered_df = instance_df.filter(pl.col("model") < pl.col("model_other"))

        winner_df = filtered_df.with_columns(
            pl.when(pl.col("score") > pl.col("score_other"))
            .then(pl.lit("model_a"))
            .when(pl.col("score") < pl.col("score_other"))
            .then(pl.lit("model_b"))
            .otherwise(pl.lit("tie"))
            .alias("winner")
        ).select(["model", "model_other", "data_instance", "winner"])
        pairwise.append(winner_df)
    pairwise = pl.concat(pairwise).rename(
        {"model": "model_a", "model_other": "model_b"}
    )
    return pairwise


def compute_tau(ranking_a, ranking_b):
    """Compute the Kendall tau between two rankings."""
    mapping = {model: i for i, model in enumerate(ranking_a)}
    indices_a = list(range(len(ranking_a)))
    indices_b = [mapping[model] for model in ranking_b]
    kendall_tau, _ = stats.kendalltau(indices_a, indices_b)
    return kendall_tau


def compute_ranking_for_subset(pairwise_results, uuids):
    """Compute the Kendall tau between the global ranking and the subset ranking."""
    pairwise_subset = pairwise_results.filter(pl.col("data_instance").is_in(uuids))
    return compute_plackett_luce_pairwise(pairwise_subset)


def compute_ranking_for_subset_dense(pairwise_results, uuids):
    """Compute the Kendall tau between the global ranking and the subset ranking."""
    pairwise_subset = pairwise_results.filter(pl.col("data_instance").is_in(uuids))
    return compute_plackett_luce_pairwise_dense(pairwise_subset)


def compute_score_ranking(results, benchmark, long=False):
    """Rank the models based on their average score."""
    if benchmark in ["helm", "leaderboard"]:
        return compute_score_ranking_dense(results, long)
    elif benchmark in ["vhelm", "lmms-eval"]:
        return compute_score_ranking_sparse(results, long)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def compute_score_ranking_dense(results, long=True):
    """Rank the models based on their average score. Assumes a complete matrix."""
    if not long:
        results = results.melt(
            id_vars=["model"], variable_name="data_instance", value_name="score"
        )
    results = results.drop_nulls()
    results = (
        results.group_by("model")
        .agg(pl.mean("score").alias("mean_score"))
        .sort("mean_score", descending=True)
    )
    return pd.Series(results["mean_score"].to_numpy(), index=results["model"])


def compute_score_ranking_sparse(results: pl.DataFrame, long=False) -> pl.DataFrame:
    """Rank the models based on their average score."""
    if not long:
        results = results.melt(
            id_vars=["model"], variable_name="data_instance", value_name="score"
        )

    results = results.with_columns(
        pl.col("data_instance").str.extract(r"(.+)_(\d+)", 1).alias("dataset")
    )
    results = results.drop_nulls("score")
    results = results.with_columns(pl.col("score").cast(pl.Float64))
    df_avg_per_dataset = results.group_by(["model", "dataset"]).agg(
        pl.col("score").mean().alias("avg_score")
    )
    df_normalized = df_avg_per_dataset.with_columns(
        (
            (pl.col("avg_score") - pl.col("avg_score").min())
            / (pl.col("avg_score").max() - pl.col("avg_score").min())
        ).alias("normalized_score")
    )

    df_final = df_normalized.group_by("model").agg(
        pl.mean("normalized_score").alias("final_score")
    )

    df_ranked = df_final.sort("final_score", descending=True)
    return pd.Series(df_ranked["final_score"].to_numpy(), index=df_ranked["model"])


def compute_mle(pairwise_results, SCALE=400, BASE=10, INIT_RATING=1000):
    """Compute the scaled Maximum Likelihood Estimate rating of each model.Thanks to the great work by Chatbot Arena"""
    if isinstance(pairwise_results, pl.DataFrame):
        pairwise_results = pairwise_results.to_pandas()

    if sum(pairwise_results["winner"] == "model_a") == 0:
        ptbl_a_win = pd.DataFrame(
            0,
            index=pairwise_results["model_a"].unique(),
            columns=pairwise_results["model_b"].unique(),
        )
    else:
        ptbl_a_win = pd.pivot_table(
            pairwise_results[pairwise_results["winner"] == "model_a"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
    if sum(pairwise_results["winner"] == "tie") == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            pairwise_results[pairwise_results["winner"] == "tie"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    if sum(pairwise_results["winner"] == "model_b") == 0:
        ptbl_b_win = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_b_win = pd.pivot_table(
            pairwise_results[pairwise_results["winner"] == "model_b"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie
    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compute_elo(pairwise_results, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    """Compute the Elo rating of each model."""
    rating = defaultdict(lambda: INIT_RATING)

    for model_a, model_b, winner in pairwise_results[
        ["model_a", "model_b", "winner"]
    ].itertuples(index=False):
        rating_a = rating[model_a]
        rating_b = rating[model_b]
        estimate_a = 1 / (1 + BASE ** ((rating_b - rating_a) / SCALE))
        estimate_b = 1 / (1 + BASE ** ((rating_a - rating_b) / SCALE))
        if winner == "model_a":
            score_a = 1
        elif winner == "model_b":
            score_a = 0
        elif winner == "tie":
            score_a = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (score_a - estimate_a)
        rating[model_b] += K * (1 - score_a - estimate_b)
    return pd.Series(rating).sort_values(ascending=False)


def compute_plackett_luce(results):
    """Rank the models using the Plackett-Luce model."""
    models = results.index.unique()
    model_to_index = {model: idx for idx, model in enumerate(models)}
    results["model_idx"] = results.index.map(model_to_index)

    rankings = []
    for _, group in results.groupby("data_instance"):
        if group["score"].dtype == "int64" and set(group["score"].unique()).issubset(
            {0, 1}
        ):
            ones = group[group["score"] == 1]["model_idx"].tolist()
            zeros = group[group["score"] == 0]["model_idx"].tolist()
            for one in ones:
                for zero in zeros:
                    rankings.append([one, zero])
        elif group["score"].dtype in ["int64", "float64"]:
            ranking = group.sort_values(by="score", ascending=False)[
                "model_idx"
            ].tolist()
            rankings.append(ranking)

    model_parameters = choix.lsr_rankings(len(model_to_index), rankings)
    return pd.Series(model_parameters, index=models).sort_values(ascending=False)


def compute_plackett_luce_pairwise(pairwise_results):
    """Rank the models using the Plackett-Luce model based on pairwise results."""
    rankings = []
    pairwise_results = pl.from_pandas(pairwise_results) if isinstance(pairwise_results, pd.DataFrame) else pairwise_results
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

    model_parameters = choix.ilsr_pairwise(len(model_to_index), rankings)

    return pd.Series(model_parameters, index=models).sort_values(ascending=False)


def compute_plackett_luce_pairwise_dense(pairwise_results):
    """Rank the models using the Plackett-Luce model based on dense pairwise results."""
    if isinstance(pairwise_results, pd.DataFrame):
        pairwise_results = pl.from_pandas(pairwise_results)

    models = (
        pl.concat([pairwise_results["model_a"], pairwise_results["model_b"]])
        .unique()
        .to_list()
    )
    model_to_index = {model: idx for idx, model in enumerate(models)}
    n_models = len(model_to_index)
    wins = np.zeros((n_models, n_models), dtype=int)

    pairwise_results = pairwise_results.with_columns(
        [
            pl.col("model_a").replace(model_to_index).cast(pl.Int64).alias("model_a"),
            pl.col("model_b").replace(model_to_index).cast(pl.Int64).alias("model_b"),
        ]
    )

    model_a_wins = (
        pairwise_results.filter(pl.col("winner") == "model_a")
        .group_by(["model_a", "model_b"])
        .agg(pl.count().alias("model_a_wins"))
    )

    model_b_wins = (
        pairwise_results.filter(pl.col("winner") == "model_b")
        .group_by(["model_b", "model_a"])
        .agg(pl.count().alias("model_b_wins"))
    )

    ties = (
        pairwise_results.filter(pl.col("winner") == "tie")
        .group_by(["model_a", "model_b"])
        .agg(pl.count().alias("ties"))
    )

    # Update the wins matrix for model_a wins
    for row in model_a_wins.iter_rows():
        model_a, model_b, count = row
        wins[model_a, model_b] += count

    # Update the wins matrix for model_b wins
    for row in model_b_wins.iter_rows():
        model_b, model_a, count = row
        wins[model_b, model_a] += count

    # Update the wins matrix for ties
    for row in ties.iter_rows():
        model_a, model_b, count = row
        wins[model_a, model_b] += count
        wins[model_b, model_a] += count

    model_parameters = choix.ilsr_pairwise_dense(wins)

    return pd.Series(model_parameters, index=models).sort_values(ascending=False)


def compute_elo_vlm(df, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    """Adhiraj's implementation of the Elo rating system for VLMs."""
    rating = defaultdict(lambda: INIT_RATING)
    unique_models = []
    for index, row in df.iterrows():
        model_pair = index.split("_vs_")
        model_a = model_pair[0]
        model_b = model_pair[1]

        if model_a not in unique_models:
            unique_models.append(model_a)
        if model_b not in unique_models:
            unique_models.append(model_b)

        for col in df.columns:
            ra = rating[model_a]
            rb = rating[model_b]
            ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
            eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))

            sa = row[col]
            rating[model_a] += K * (sa - ea)
            rating[model_b] += K * (1 - sa - eb)

    return rating


def pretty_print_model_ratings(ratings):
    df = (
        pd.DataFrame(
            [[n, ratings[n]] for n in ratings.keys()], columns=["Model", "Elo rating"]
        )
        .sort_values("Elo rating", ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1
    return df


def pretty_print_two_ratings(ratings_1, ratings_2, column_names):
    df = (
        pd.DataFrame(
            [[n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()],
            columns=["Model", column_names[0], column_names[1]],
        )
        .sort_values(column_names[0], ascending=False)
        .reset_index(drop=True)
    )
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df
