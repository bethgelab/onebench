from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import os
# from io import load_arena_results, load_helm_results_pairwise
from elo import compute_mle, to_pairwise
import polars as pl
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
import requests
import choix
from utils import generate_top1_list

def load_long(df):
    return df.melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )


def load_helm_results_pairwise():
    """Load or compute the pairwise model comparisons."""
    pairwise_results_path = "data/llm/helm/pairwise.parquet"
    results = pl.read_parquet(f"data/llm/helm/results.parquet")
    results_long = load_long(results)
    pairwise_results = to_pairwise(results_long,long=True)
    pairwise_results.write_parquet(pairwise_results_path)
    return pairwise_results


def load_arena_results():
    """Load the LLM evaluation results."""
    path = "/Users/heikekoenig/irp/lifelong_analysis/data/llm/arena/arena/results.json"
    if not os.path.exists(path):
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

def compute_pl_pairwise(pairwise_results):
    """Rank the models using the Plackett-Luce model based on pairwise results."""
    rankings = []

    models = pl.concat([pairwise_results["model_a"], pairwise_results["model_b"]]).unique().to_list()
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
    model_parameters = choix.lsr_pairwise(len(model_to_index), rankings)

    return pd.Series(model_parameters, index=models).sort_values(ascending=False)

def match_models(helm_models, arena_models):
    """Match models from the helm results to the arena results."""
    exclude = [
        "anthropic_claude-2.0",
        "anthropic_claude-2.1",
        "anthropic_claude-3-haiku-20240307",
        "anthropic_claude-3-sonnet-20240229",
        "openai_gpt-4-0613",
        "openai_gpt-3.5-turbo-0613",
        "anthropic_claude-instant-1.2",
        "anthropic_claude-instant-v1",
        "allenai_olmo-7b",
        "olmo-7b-instruct",
        # "mistralai_mistral-large-2402"
    ]
    helm_to_arena = {}
    for model in helm_models:
        if model in exclude:
            continue
        model_name = "_".join(model.split("_")[1:])
        match, score, _ = process.extractOne(
            model_name, arena_models, scorer=fuzz.WRatio
        )
        print(f"{model} -> {match} ({score})")
        if score > 95:
            helm_to_arena[model] = match
    helm_to_arena.update(
        {
            "google_gemini-1.0-pro-001": "gemini-1.5-pro-api-0514",
            "meta_llama-3-70b": "llama-3-70b-instruct",
            "mistralai_mistral-medium-2312": "mistral-medium",
            "qwen_qwen1.5-72b": "qwen1.5-72b-chat",
            "01-ai_yi-34b": "yi-34b-chat",
            # "allenai_olmo-7b": "olmo-7b-instruct",
            "databricks_dbrx-instruct": "dbrx-instruct-preview",
        }
    )
    for helm_model, arena_model in helm_to_arena.items():
        assert helm_model in helm_models and arena_model in arena_models
    return helm_to_arena

def get_models(df):
    """Get unique models from the results dataframe."""
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    return models.tolist()

def illustrate_ranking_change(args):
    """Illustrate the change in ranking when combining HELM and Arena results."""
    results_pairwise_helm = load_helm_results_pairwise().to_pandas()
    results_pairwise_arena = load_arena_results().to_pandas()

    results_pairwise_helm = results_pairwise_helm[["model_a", "model_b", "winner"]]
    results_pairwise_arena = results_pairwise_arena[["model_a", "model_b", "winner"]]

    helm_models = get_models(results_pairwise_helm)
    arena_models = get_models(results_pairwise_arena)
    helm_to_arena = match_models(helm_models, arena_models)

    # rename models in the helm results
    results_pairwise_helm["model_a"] = results_pairwise_helm["model_a"].apply(
        lambda x: helm_to_arena.get(x, x)
    )
    results_pairwise_helm["model_b"] = results_pairwise_helm["model_b"].apply(
        lambda x: helm_to_arena.get(x, x)
    )

    # compute MLE for Arena
    results_pairwise_arena = results_pairwise_arena[
        results_pairwise_arena["model_a"].isin(helm_to_arena.values())
        & results_pairwise_arena["model_b"].isin(helm_to_arena.values())
    ]
    ranking_arena = compute_pl_pairwise(pl.from_pandas(results_pairwise_arena))
    # compute joint MLE for Helm and Arena
    results_pairwise_helm = results_pairwise_helm[
        results_pairwise_helm["model_a"].isin(helm_to_arena.values())
        & results_pairwise_helm["model_b"].isin(helm_to_arena.values())
    ]
    results_pairwise_joint = pd.concat([results_pairwise_helm, results_pairwise_arena])
    ranking_joint = compute_pl_pairwise(pl.from_pandas(results_pairwise_joint))
    matched_keys = set(helm_to_arena)
    helm = pd.read_parquet('data/llm/helm/results.parquet')
    helm = helm.set_index("model",drop=True)
    filtered_helm = helm[helm.index.isin(matched_keys)]
    helm_rank = compute_pl_pairwise(filtered_helm)
    print()
    print("Only arena")
    print(ranking_arena)
    print()
    print("Only HELM")
    print(helm_rank)
    print()
    print("Joint")
    print(ranking_joint)
    print()

def main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    illustrate_ranking_change(args)

