"""Loading and saving results."""

from pathlib import Path
import polars as pl
import pandas as pd
import json
import numpy as np
from itertools import zip_longest
from analysis import (
    to_pairwise,
    compute_plackett_luce_pairwise_dense,
)
import requests


def get_benchmark_path(benchmark):
    data_dir = Path(__file__).parents[2] / "data"
    if benchmark in ["helm", "leaderboard", "all", "synthetic"]:
        return data_dir / "llm" / benchmark
    if benchmark in ["vhelm", "lmms-eval"]:
        return data_dir / "vlm" / benchmark
    raise ValueError(f"Invalid benchmark: {benchmark}")


def load_instances(benchmark="all"):
    """Load the data instances for the specified benchmark."""
    instances_path = get_benchmark_path(benchmark) / "instances.json"
    with open(instances_path, "r") as file:
        return json.load(file)


def load_results(benchmark="all"):
    """Load the LLM evaluation results for the specified benchmark."""
    results_path = get_benchmark_path(benchmark) / "results.parquet"
    return pl.read_parquet(results_path)


def load_long_results(benchmark="all"):
    """Load the LLM evaluation results in the long format for the specified benchmark."""
    results = load_results(benchmark)
    return results.melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )


def load_pairwise_results(benchmark="all"):
    """Load or compute the pairwise model comparisons for the specified benchmark."""
    pairwise_results_path = get_benchmark_path(benchmark) / "pairwise.parquet"
    if pairwise_results_path.exists():
        pairwise_results = pl.read_parquet(pairwise_results_path)
    else:
        pairwise_results = to_pairwise(load_long_results(benchmark))
        pairwise_results.write_parquet(pairwise_results_path)
    return pairwise_results


def load_filtered_vhelm_results():
    """Filter out the rows with missing data from the VHELM results."""
    results = load_results("vhelm")
    rows_to_drop = [5, 6, 12, 13, 14, 15, 16, 17, 19, 21]
    mask = ~pl.Series(range(results.height)).is_in(rows_to_drop)
    filtered_df = results.filter(mask)
    nulls_per_column = filtered_df.null_count().row(0)
    columns_with_no_nulls = [
        col
        for col, null_count in zip(results.columns, nulls_per_column)
        if null_count == 0
    ]
    return filtered_df.select(columns_with_no_nulls)


def load_or_compute_global_ranking(benchmark="all"):
    """Load or compute the global ranking for the specified benchmark."""
    ranking_path = get_benchmark_path(benchmark) / "ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path, index_col=0)
    else:
        pairwise = load_pairwise_results(benchmark)
        ranking = compute_plackett_luce_pairwise_dense(pairwise)
        ranking.to_frame().to_csv(ranking_path)
    return ranking


def load_embeddings(benchmark, type="question"):
    """Load embeddings for the data instances of the specified benchmark."""
    embeddings_path = get_benchmark_path(benchmark) / f"{type}_embeddings.npy"
    embeddings = np.load(embeddings_path)
    return embeddings


def format_data_instance(instance, labels=None, json=False):
    """Format the data instance for display or embedding."""
    question = instance["question"]["text"]
    references = [reference["text"] for reference in instance["references"]]
    if labels is not None:
        references = [
            f"{label}. {reference}"
            for label, reference in zip_longest(labels, references)
        ]
    if not json:
        return question + "\n" + "\n".join(references)
    return question + " " + " ".join(references)


def load_arena_results():
    """Load the LLM evaluation results for the Arena benchmark."""
    path = get_benchmark_path("arena") / "results.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240527_public.json"
        response = requests.get(url)
        with open(path, "wb") as file:
            file.write(response.content)

    with open(path, "r") as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    battles = battles[battles["anony"]]
    battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    return pl.DataFrame(battles)


def load_synthetic_model_parameters():
    """Load the synthetic model parameters."""
    path = get_benchmark_path("synthetic") / "model_parameters.csv"
    return pd.read_csv(path, index_col=0)
