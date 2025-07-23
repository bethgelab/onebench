import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from logger import Logger
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():
    parser = argparse.ArgumentParser(description="Correlation Stats")

    parser.add_argument(
        "--source_dir",
        default="/Users/heikekoenig/irp/lifelong_analysis/data/",
        help="root directory to load results",
        type=str,
    )

    parser.add_argument("--metric", default="numeric", help="binary/numeric", type=str)

    parser.add_argument(
        "--dataset",
        default="lmms-eval",
        help="lmms-eval/vhelm/leaderboard/helm",
        type=str,
    )

    args = parser.parse_args()

    if args.dataset == "vhelm" or args.dataset == "lmms-eval":
        domain = "vlm"
    else:
        domain = "llm"

    path_src = f"{args.source_dir}/{domain}"
    sys.stdout = Logger(
        os.path.join(f"results/correlation/{domain}_{args.dataset}.log")
    )

    if args.dataset == "vhelm" or args.dataset == "lmms-eval":
        df = pd.read_parquet(f"{path_src}/{args.dataset}/{args.metric}.parquet")
        save_dir = f"results/correlation/{args.dataset}_{args.metric}"

    else:
        df = pd.read_parquet(f"{path_src}/{args.dataset}/results.parquet")
        save_dir = f"results/correlation/{args.dataset}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # df = pd.read_parquet(f"{args.source_dir}/results.parquet")
    if args.dataset == "helm" or args.dataset == "leaderboard":
        df.set_index("model", inplace=True)

    print("Number of columns ", len(df.columns))

    pearson_corr = df.T.corr(method="pearson")
    print("computed pearson correlation")

    spearman_corr = df.T.corr(method="spearman")
    print("computed spearman correlation")

    # we only need the lower triangle of the matrix, upper triangle is the same
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 24))
    #
    # sns.heatmap(pearson_corr, mask=mask, ax=axes[0], annot=True, cmap='coolwarm', vmin=pearson_corr.min().min(), vmax=pearson_corr.max().max())
    # axes[0].set_title('Pearson Correlation')
    #
    # sns.heatmap(spearman_corr, mask=mask, ax=axes[1], annot=True, cmap='coolwarm', vmin=spearman_corr.min().min(), vmax=spearman_corr.max().max())
    # axes[1].set_title('Spearman Correlation')
    #
    # plt.savefig(f"{save_dir}/correlation_{args.dataset}.png")

    # Check for nonlinear relationship using polynomial regression

    model1_performance = df.iloc[0, :].values
    model2_performance = df.iloc[1, :].values

    print(df.iloc[0, :])

    X = model1_performance.reshape(-1, 1)
    y = model2_performance

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    y_pred = lin_reg.predict(X_poly)

    plt.scatter(model1_performance, model2_performance, color="blue")
    plt.plot(model1_performance, y_pred, color="red")
    plt.xlabel("Model 1 Performance")
    plt.ylabel("Model 2 Performance")
    plt.title("Nonlinear Relationship Analysis")
    plt.savefig(f"{save_dir}/perf_regression.png")


if __name__ == "__main__":
    main()
