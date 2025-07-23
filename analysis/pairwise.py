import pandas as pd
import os
import choix
import networkx as nx
import argparse
import numpy as np
from matplotlib import pyplot as plt
from logger import Logger
import sys
from itertools import combinations

def main():
    parser = argparse.ArgumentParser(description="Bradley-Terry Analysis")

    parser.add_argument(
        "--source_dir", default="/Users/heikekoenig/irp/lifelong_analysis/data/vlm/",
        help="root directory to load results", type=str)

    parser.add_argument(
        "--save_graph", default="/Users/heikekoenig/irp/lifelong_analysis/results/bradley-terry/",
        help="root directory to load results", type=str)

    parser.add_argument(
        "--dataset", default="vhelm",
        help="lmms-eval/vhelm", type=str)

    parser.add_argument(
        "--metric", default="binary",
        help="binary/numeric", type=str)

    parser.add_argument(
        "--tie",  action='store_true')

    parser.add_argument(
        "--alpha", default=0.0001,
        help="alpha value for regularisation", type=float)


    args = parser.parse_args()
    path_src = f"{args.source_dir}/{args.dataset}/{args.metric}.parquet"
    df = pd.read_parquet(path_src)

    if args.tie:
        sys.stdout = Logger(os.path.join(f'results/bradley-terry/tie_{args.dataset}_{args.metric}_{args.alpha}.log'))
    else:
        sys.stdout = Logger(os.path.join(f'results/bradley-terry/no_tie_{args.dataset}_{args.metric}_{args.alpha}.log'))

    if args.dataset == "vhelm":
        df.set_index(df.columns[0], inplace=True)
    comparison_outcomes = create_mapping(args, df)

    n_items = len(df.index)
    model_to_index = {model: idx for idx, model in enumerate(df.index)}
    index_to_model = {idx: model for model, idx in model_to_index.items()}

    #make a directed graph
    graph = nx.DiGraph()
    graph.add_edges_from(comparison_outcomes)
    graph = nx.relabel_nodes(graph, index_to_model)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  # Layout for visualization
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold',
            arrowsize=20)

    # Save the graph as a PNG file
    plt.savefig(f"{args.save_graph}/{args.dataset}_{args.metric}.png", format='png')
    # plt.show()

    weights = choix.ilsr_pairwise(n_items, comparison_outcomes, alpha=args.alpha)
    ranking = np.argsort(weights)[::-1]

    # Print the ranking
    print("Bradley-Terry Ranking:")
    for i in ranking:
        print(df.index[i])

    print()
    print("Probabilities of wins")
    for i in range(n_items):
        for j in range(n_items):
            print(f"{df.index[i]} vs {df.index[j]}: {choix.probabilities([i,j], weights)[0]}")
        print()

    # # Compute Kendall Tau distances between all pairs of models
    # model_pairs = list(combinations(range(len(weights)), 2))
    # kendall_tau_distances = {}
    # for i, j in model_pairs:
    #     distance = choix.kendalltau_dist(weights[i], weights[j])
    #     kendall_tau_distances[(i, j)] = distance
    #
    # # Convert index to model names
    # # index_to_model = {v: k for k, v in model_to_index.items()}
    # kendall_tau_distances_named = {(index_to_model[i], index_to_model[j]): dist for (i, j), dist in
    #                                kendall_tau_distances.items()}
    #
    # print("Kendall Tau distances between model pairs:")
    # for models, distance in kendall_tau_distances_named.items():
    #     print(f"{models}: {distance}")

def create_mapping(args, df):
    model_to_index = {model: idx for idx, model in enumerate(df.index)}
    comparison_outcomes = []

    for sample in df.columns:
        sample_data = df[sample]

        # important to drop None values
        sample_data = sample_data.dropna()

        sample_values = sample_data.values
        model_indices = [model_to_index[model] for model in sample_data.index]

        # Generate comparisons
        if args.tie:
            for i in range(len(sample_values)):
                for j in range(len(sample_values)):
                    if i == j:
                        continue
                    if sample_values[i] > sample_values[j]:
                        comparison_outcomes.append((model_indices[i], model_indices[j]))
                    elif sample_values[i] < sample_values[j]:
                        comparison_outcomes.append((model_indices[j], model_indices[i]))
                    else:  # sample_values[i] == sample_values[j]
                        # we are representing ties by adding both (i, j) and (j, i)
                        comparison_outcomes.append((model_indices[i], model_indices[j]))
                        comparison_outcomes.append((model_indices[j], model_indices[i]))
        else:
            for i in range(len(sample_values)):
                for j in range(len(sample_values)):
                    if i != j and sample_values[i] > sample_values[j]:
                        comparison_outcomes.append((model_indices[i], model_indices[j]))

    print(len(comparison_outcomes))
    return comparison_outcomes

def create_mapping_score(args, df):
    model_to_index = {model: idx for idx, model in enumerate(df.index)}
    comparison_outcomes = []

    for sample in df.columns:
        sample_data = df[sample]

        # important to drop None values
        sample_data = sample_data.dropna()

        sample_values = sample_data.values
        model_indices = [model_to_index[model] for model in sample_data.index]

        # Generate comparisons
        if args.tie:
            for i in range(len(sample_values)):
                for j in range(i + 1, len(sample_values)):
                    if sample_values[i] > sample_values[j]:
                        comparison_outcomes.append((model_indices[i], model_indices[j], 1))
                        comparison_outcomes.append((model_indices[j], model_indices[i], 0))
                    elif sample_values[i] < sample_values[j]:
                        comparison_outcomes.append((model_indices[j], model_indices[i], 1))
                        comparison_outcomes.append((model_indices[i], model_indices[j], 0))
                    else:  # Tie case
                        comparison_outcomes.append((model_indices[i], model_indices[j], 0.5))
                        comparison_outcomes.append((model_indices[j], model_indices[i], 0.5))  # Symmetric tie
        else:
            for i in range(len(sample_values)):
                for j in range(len(sample_values)):
                    if i != j and sample_values[i] > sample_values[j]:
                        comparison_outcomes.append((model_indices[i], model_indices[j], 1))
                        comparison_outcomes.append((model_indices[j], model_indices[i], 0))

    print(len(comparison_outcomes))
    return comparison_outcomes

if __name__ == "__main__":
    main()