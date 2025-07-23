import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(
    num_models=100, num_samples=10000, distribution="gaussian", dispersion=0.25
):
    if distribution == "gaussian":
        means = np.random.uniform(0, 1, num_models)
        stds = np.random.uniform(0.0, dispersion, num_models)
        sorted_indices = np.argsort(means)[::-1]
        means = means[sorted_indices]
        stds = stds[sorted_indices]
        model_params = pd.DataFrame(
            {
                "model": [f"model{i+1}" for i in range(num_models)],
                "means": means,
                "stds": stds,
            }
        )
    else:
        locations = dispersion * np.arange(1, num_models + 1)
        scales = np.ones(num_models)
        sorted_indices = np.argsort(locations)[::-1]
        locations = locations[sorted_indices]
        scales = scales[sorted_indices]
        model_params = pd.DataFrame(
            {
                "model": [f"model{i+1}" for i in range(num_models)],
                "locations": locations,
                "scales": scales,
            }
        )

    data = []
    sample_types = ["binary", "numerical"]
    for _ in range(num_samples):
        sample_type = np.random.choice(sample_types)

        if distribution == "gaussian":
            scores = np.random.normal(means, stds)
        else:
            scores = np.random.gumbel(locations, scales)

        if sample_type == "binary":
            min_score = np.min(scores)
            max_score = np.max(scores)
            threshold = np.random.uniform(min_score, max_score)
            scores = (scores > threshold).astype(float)
        data.append(scores)

    data = np.array(data).T
    df = pd.DataFrame(
        data,
        index=[f"model{i+1}" for i in range(num_models)],
        columns=[f"sample{i+1}" for i in range(num_samples)],
    )
    df = df.sample(frac=1)
    df = df.reset_index().rename(columns={"index": "model"})

    output_dir = Path("data/llm/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_dir / "results.parquet")
    model_params.to_csv(output_dir / "model_parameters.csv", index=False)

    return df, model_params


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic measurements")
    parser.add_argument(
        "--num_models", type=int, default=100, help="Number of models to generate"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["gaussian", "gumbel"],
        default="gaussian",
        help="Distribution to use for generating scores (gaussian or gumbel)",
    )
    parser.add_argument(
        "--dispersion",
        type=float,
        default=0.1,
        help="Dispersion parameter (max std for Gaussian, scaling factor for Gumbel means)",
    )

    args = parser.parse_args()
    generate_synthetic_data(
        num_models=args.num_models,
        num_samples=args.num_samples,
        distribution=args.distribution,
        dispersion=args.dispersion,
    )


if __name__ == "__main__":
    main()
