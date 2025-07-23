from argparse import ArgumentParser
import pandas as pd
import os
import polars as pl
# from analysis.io import load_leaderboard_results
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    root = Path(__file__).parent.parent

    parser = ArgumentParser()
    parser.add_argument("--src_dir", type=str, default= "data/")
    parser.add_argument("--out_dir", type=str, default= "results/hamming/")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument("--benchmark", type=str, default='llm')

    args = parser.parse_args()
    print(args.src_dir)
    measure_model_correlation(args)


def measure_model_correlation(args):

    if args.benchmark == 'llm':
        file_path = args.src_dir + "llm/leaderboard/results.parquet"
        data = pl.read_parquet(file_path).to_pandas()
        data.set_index("model", inplace=True)

    else:
        file_path_bin = args.src_dir + "vlm/lmms-eval/binary.parquet"
        file_path_num = args.src_dir + "vlm/lmms-eval/numeric.parquet"
        data_bin = pl.read_parquet(file_path_bin).to_pandas()
        data_bin.set_index("model", inplace=True)
        data_num = pl.read_parquet(file_path_num).to_pandas()
        data_num.set_index("model", inplace=True)
        data = pd.merge(data_bin, data_num, left_index=True, right_index=True, how='outer')
        data.to_parquet('data/vlm/vhelm/results.parquet')
        # file_path = args.src_dir + "vlm/vhelm/results.parquet"
        # data = pd.read_parquet(file_path)


    print(data)
    data["accuracy"] = data.mean(axis=1, skipna=True)
    data = data.sort_values(by="accuracy", ascending=False)
    data = data.drop(columns="accuracy").head(args.num_models)

    if args.benchmark == 'llm':
        distance_matrix = pdist(data, metric="hamming")
        # distance_matrix = squareform(distance_matrix)
        distance_matrix = 1 - squareform(distance_matrix)
        distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)
    else:
        # distance_matrix = pdist(data, metric="hamming")
        # # distance_matrix = squareform(distance_matrix)
        # distance_matrix = 1 - squareform(distance_matrix)
        # distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)

        n_models = data.shape[0]
        distance_matrix = np.ones((n_models,n_models))
        for i in range(n_models):
            for j in range(i + 1, n_models):
                valid_indices_names = data.iloc[i].notna() & data.iloc[j].notna()
                valid_indices_names = valid_indices_names[valid_indices_names].index.tolist()
                valid_indices = [data.columns.get_loc(col) for col in valid_indices_names]
                if len(valid_indices) > 0:
                    hamming_dist = np.mean(data.iloc[i, valid_indices].values != data.iloc[j, valid_indices].values)
                else:
                    hamming_dist = np.nan
                distance_matrix[i, j] = 1-hamming_dist
                distance_matrix[j, i] = 1-hamming_dist

            # Convert to DataFrame
        # distance_matrix = 1 - squareform(distance_matrix)
        distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)

    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(distance_df, cmap="viridis", fmt=".2f")
    ax.set_aspect("equal")
    plt.title("Hamming distance between the LMMs-Eval models")

    os.makedirs(args.out_dir, exist_ok=True)
    file_name = args.out_dir + f"{'leaderboard' if args.benchmark == 'llm' else 'lmms_eval'}_pairwise_hamming_distance.pdf"
    fig.savefig(file_name, bbox_inches="tight")


if __name__ == "__main__":
    main()
