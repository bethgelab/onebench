import pandas as pd
import os
import choix
import argparse
import numpy as np
from logger import Logger
import sys
import random
from pathlib import Path


def sample_rankings(df, model_to_index):
    rankings = []
    for sample in df.columns:
        sample_results = df[sample].dropna()  # Drop None values
        ones = sample_results[sample_results == 1].index.tolist()
        ones_indices = [model_to_index[model] for model in ones]
        zeros = sample_results[sample_results == 0].index.tolist()
        zeros_indices = [model_to_index[model] for model in zeros]

        # Create ranking list for each model with score of 1
        for one in ones_indices:
            ranking = [one]
            if zeros:
                ranking.append(list(zeros_indices))
            rankings.append(ranking)

    return rankings


def main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Plackett Luce Analysis")
    parser.add_argument(
        "--source_dir",
        default=root / "data/vlm",
        help="root directory to load results",
        type=Path,
    )
    parser.add_argument(
        "--save_graph",
        default=root / "results/plackett-luce",
        help="root directory to load results",
        type=Path,
    )
    parser.add_argument(
        "--rank", default="sample", choices=["random", "benchmark", "sample"], type=str
    )
    parser.add_argument(
        "--alpha", default=0.01, help="Alpha value for regularisation", type=float
    )
    parser.add_argument(
        "--dataset", default="vhelm", choices=["llms-eval", "vhelm"], type=str
    )
    parser.add_argument(
        "--metric", default="binary", choices=["binary", "numeric"], type=str
    )
    parser.add_argument("--tie", action="store_true")

    args = parser.parse_args()
    path_src = f"{args.source_dir}/{args.dataset}/{args.metric}.parquet"
    df = pd.read_parquet(path_src)

    sys.stdout = Logger(
        os.path.join(
            f"results/plackett-luce/{args.dataset}_{args.metric}_{args.alpha}_{args.rank}.log"
        )
    )

    if args.dataset == "vhelm":
        df.set_index(df.columns[0], inplace=True)

    rankings = []
    model_to_index = {model: idx for idx, model in enumerate(df.index)}

    if args.rank == "random":
        no_samples = 100
        column_names = list(df.columns)
        random.shuffle(column_names)

        # groups of 100. Subject to change. This gives us more rankings to work with.
        column_groups = [
            column_names[i : i + no_samples]
            for i in range(0, len(column_names), no_samples)
        ]

        for group in column_groups:
            dataset_results = df[group].mean(axis=1, skipna=True).dropna()
            # dataset_results = df[group].sum(axis=1, skipna=True)

            # # Filter out models with a sum of zero.
            # dataset_results = dataset_results[dataset_results != 0]
            print(dataset_results)
            ranked_models = dataset_results.sort_values(ascending=False).index.tolist()
            ranked_indices = [model_to_index[model] for model in ranked_models]
            rankings.append(ranked_indices)

    elif args.rank == "benchmark":
        samples_dict = {}
        for sample in df.columns:
            dataset = "_".join(sample.split("_")[:-1])
            if dataset not in samples_dict:
                samples_dict[dataset] = []
            samples_dict[dataset].append(sample)

        for dataset, samples in samples_dict.items():
            dataset_results = df[samples].sum(axis=1, skipna=True)
            dataset_results = dataset_results[dataset_results != 0]
            print(dataset)
            print(dataset_results)
            ranked_models = dataset_results.sort_values(ascending=False).index.tolist()
            ranked_indices = [model_to_index[model] for model in ranked_models]
            rankings.append(ranked_indices)

    elif args.rank == "sample":
        rankings = sample_rankings(df, model_to_index)

    # Use choix.ilsr_ranking to generate model rankings
    num_models = len(df.index)
    params = np.zeros(num_models)
    weights = choix.ilsr_top1(
        num_models, rankings, alpha=args.alpha, initial_params=params
    )
    # weights = choix.ilsr_rankings(num_models, rankings, alpha=args.alpha)

    # Display the estimated skill parameters
    model_skill = {model: weights[idx] for model, idx in model_to_index.items()}
    sorted_model_skill = sorted(model_skill.items(), key=lambda x: x[1], reverse=True)

    print("Estimated model skill parameters:")
    for model, skill in sorted_model_skill:
        print(f"{model}: {skill}")


if __name__ == "__main__":
    main()
