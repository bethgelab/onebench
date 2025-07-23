from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from analysis import load_arena_results, load_pairwise_results
from analysis import compute_plackett_luce_pairwise_dense
from rapidfuzz import fuzz, process

ROOT = Path(__file__).parent.parent


def main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    illustrate_ranking_change(args)


def illustrate_ranking_change(args):
    """Illustrate the change in ranking when combining HELM and Arena results."""
    pairwise_results_helm = load_pairwise_results(benchmark="helm")
    ranking_helm = compute_plackett_luce_pairwise_dense(pairwise_results_helm)

    pairwise_results_helm = pairwise_results_helm.to_pandas()
    pairwise_results_arena = load_arena_results().to_pandas()

    pairwise_results_helm = pairwise_results_helm[["model_a", "model_b", "winner"]]
    pairwise_results_arena = pairwise_results_arena[["model_a", "model_b", "winner"]]

    helm_models = get_models(pairwise_results_helm)
    arena_models = get_models(pairwise_results_arena)
    helm_to_arena = match_models(helm_models, arena_models)

    # rename models in the helm results
    pairwise_results_helm["model_a"] = pairwise_results_helm["model_a"].apply(
        lambda x: helm_to_arena.get(x, x)
    )
    pairwise_results_helm["model_b"] = pairwise_results_helm["model_b"].apply(
        lambda x: helm_to_arena.get(x, x)
    )

    # compute MLE for Arena
    pairwise_results_arena = pairwise_results_arena[
        pairwise_results_arena["model_a"].isin(helm_to_arena.values())
        & pairwise_results_arena["model_b"].isin(helm_to_arena.values())
    ]
    ranking_arena = compute_plackett_luce_pairwise_dense(pairwise_results_arena)

    # compute joint MLE for Helm and Arena
    pairwise_results_helm = pairwise_results_helm[
        pairwise_results_helm["model_a"].isin(helm_to_arena.values())
        & pairwise_results_helm["model_b"].isin(helm_to_arena.values())
    ]
    results_pairwise_joint = pd.concat([pairwise_results_helm, pairwise_results_arena])
    ranking_joint = compute_plackett_luce_pairwise_dense(results_pairwise_joint)

    print(ranking_helm, "\n")
    print(ranking_arena, "\n")
    print(ranking_joint)


def get_models(df):
    """Get unique models from the results dataframe."""
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    return models.tolist()


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
            "allenai_olmo-7b": "olmo-7b-instruct",
            "databricks_dbrx-instruct": "dbrx-instruct-preview",
        }
    )
    for helm_model, arena_model in helm_to_arena.items():
        assert helm_model in helm_models and arena_model in arena_models
    return helm_to_arena


if __name__ == "__main__":
    main()
