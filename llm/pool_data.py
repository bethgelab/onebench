import argparse
from pathlib import Path
import tqdm
import polars as pl
import json

from analysis import (
    load_instances,
    load_results,
)


def _main():
    root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Where to save the result",
        default=root / "data/llm",
    )
    args = parser.parse_args()
    pool_data(args)


def pool_data(args):
    """Pool the data from the HELM and Leaderboard benchmarks."""
    helm_samples = load_instances(benchmark="helm")
    helm_df = load_results(benchmark="helm")
    leaderboard_samples = load_instances(benchmark="leaderboard")
    leaderboard_df = load_results(benchmark="leaderboard")

    uuid_mapping = find_duplicate_samples(helm_samples, leaderboard_samples)
    leaderboard_df = leaderboard_df.rename(mapping=uuid_mapping)
    joint_samples = {
        **helm_samples,
        **{k: v for k, v in leaderboard_samples.items() if k not in uuid_mapping},
    }

    # lowercase model names
    helm_df = helm_df.with_columns(pl.col("model").str.to_lowercase().alias("model"))
    leaderboard_df = leaderboard_df.with_columns(
        pl.col("model").str.to_lowercase().alias("model")
    )

    # rename models to match the leaderboard
    helm_to_leaderboard = {
        "01-ai_yi-34b": "01-ai/yi-34b-200k",
        "cohere_command-r-plus": "cohereforai/c4ai-command-r-plus",
        "google_gemma-7b": "google/gemma-7b",
        "meta_llama-3-70b": "meta-llama/meta-llama-3-70b",
        "meta_llama-3-8b": "meta-llama/meta-llama-3-8b",
        "microsoft_phi-2": "microsoft/phi-2",
        "mistralai_mixtral-8x22b": "mistral-community/mixtral-8x22b-v0.1",
        "qwen_qwen1.5-14b": "qwen/qwen1.5-14b",
        "qwen_qwen1.5-7b": "qwen/qwen1.5-7b",
    }
    helm_df = helm_df.with_columns(
        pl.col("model").replace(helm_to_leaderboard).alias("model")
    )

    joint_df = helm_df.join(leaderboard_df, on="model", how="full", coalesce=True)
    duplicate_columns = find_duplicate_columns(joint_df)
    joint_df = resolve_all_duplicates(joint_df, duplicate_columns)

    path = args.data_dir / "all"
    path.mkdir(parents=True, exist_ok=True)
    joint_df.write_parquet(path / "results.parquet")
    with open(path / "instances.json", "w") as f:
        json.dump(joint_samples, f, indent=4, sort_keys=True)

    print("Dataframes merged and saved successfully with duplicates resolved.")


def find_duplicate_samples(helm_samples, leaderboard_samples):
    """Find samples that are duplicated in the HELM and Leaderboard benchmarks."""
    uuid_mapping = {}
    for helm_uuid, helm_sample in helm_samples.items():
        for leaderboard_uuid, leaderboard_sample in leaderboard_samples.items():
            if helm_sample == leaderboard_sample:
                uuid_mapping[leaderboard_uuid] = helm_uuid
    return uuid_mapping


def find_duplicate_columns(joint_df):
    """Find columns that are duplicated in the joint dataframe."""
    duplicate_columns = []
    for col in joint_df.columns:
        if col.endswith("_right"):
            original_col = col.replace("_right", "")
            if original_col in joint_df.columns:
                duplicate_columns.append((original_col, col))
    return duplicate_columns


def resolve_all_duplicates(df, col_pairs):
    """Resolve all duplicated columns in the dataframe."""
    expressions = []
    cols_to_drop = []
    for col1, col2 in tqdm.tqdm(col_pairs):

        resolved_col = (
            pl.when(pl.col(col1).is_null() & ~pl.col(col2).is_null())
            .then(pl.col(col2))
            .when(~pl.col(col1).is_null())
            .then(pl.col(col1))
            .otherwise(None)
            .alias(col1)
        )
        expressions.append(resolved_col)
        cols_to_drop.append(col2)

    df = df.with_columns(expressions)
    df = df.drop(cols_to_drop)

    return df


def count_agreements(df, col_pairs):
    """Count the number of agreements between duplicated columns in the dataframe."""

    total_both_not_null = 0
    total_agreements = 0
    agreed_models = set()

    for col1, col2 in col_pairs:
        agreement = df.select(pl.col(col1) == pl.col(col2))
        total_agreements += agreement.sum().item()
        total_both_not_null += agreement.height - agreement.null_count().item()
        for model in df.filter(agreement[col1].is_not_null())["model"]:
            agreed_models.add(model)

    agreement_stats = {
        "total_cases_with_both_values": total_both_not_null,
        "total_agreements": total_agreements,
        "agreement_percentage": total_agreements / total_both_not_null * 100,
        "agreed_models": agreed_models,
    }

    return agreement_stats


if __name__ == "__main__":
    _main()
