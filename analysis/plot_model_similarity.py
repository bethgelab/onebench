from argparse import ArgumentParser
import pandas as pd
from analysis.io import load_leaderboard_results, load_helm_results
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "figures")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument(
        "--benchmark", type=str, choices=["helm", "leaderboard"], default="leaderboard"
    )
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()
    plot_model_similarity(args)


def plot_model_similarity(args):
    if args.benchmark == "helm":
        data = load_helm_results().to_pandas()
    elif args.benchmark == "leaderboard":
        data = load_leaderboard_results().to_pandas()

    data.set_index("model", inplace=True)
    data["accuracy"] = data.mean(axis=1)
    data = data.sort_values(by="accuracy", ascending=False)
    data = data.drop(columns="accuracy").head(args.num_models)
    print(f"Loaded {len(data)} models")

    # filter out non-binary integer columns
    binary_integer_cols = [
        col for col in data.columns if set(data[col].unique()).issubset({0, 1})
    ]
    data = data[binary_integer_cols]
    print(f"Using {len(data.columns)} binary integer columns")

    distance_matrix = pdist(data, metric="hamming")
    distance_matrix = 1 - squareform(distance_matrix)
    distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)

    distance_df.columns = distance_df.columns.str.pad(50, side="left")
    distance_df.index = distance_df.index.str.pad(50, side="left")

    fig, ax = plt.subplots(figsize=(15, 15))
    if args.labels:
        monospace_font = FontProperties(family="monospace")
        for label in ax.get_xticklabels():
            label.set_fontproperties(monospace_font)

        for label in ax.get_yticklabels():
            label.set_fontproperties(monospace_font)
    else:
        ax.tick_params(left=False, bottom=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    sns.heatmap(
        distance_df,
        cmap="viridis",
        fmt=".2f",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8, "aspect": 30},
    )
    ax.set_aspect("equal")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.out_dir / f"similarity_{args.benchmark}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
