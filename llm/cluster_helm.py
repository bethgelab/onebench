import polars as pl
from argparse import ArgumentParser
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text


def _main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "results/helm/results.parquet",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "img",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)
    cluster_helm(args)


def cluster_helm(args):
    """Cluster the HELM data."""
    df = pl.read_parquet(args.input)
    model_names = [name.split("_")[-1] for name in df["model"]]
    df = df.drop("model")

    kmeans = KMeans(n_clusters=6)
    clusters = kmeans.fit_predict(df)
    fig = plot_clusters(df, clusters, labels=model_names)
    fig.savefig(args.out_dir / "models_clustered.png", bbox_inches="tight", dpi=300)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(df.transpose())
    fig = plot_clusters(df.transpose(), clusters)
    fig.savefig(args.out_dir / "tasks_clustered.png")


def plot_clusters(df, clusters, labels=None, figsize=(12, 5)):
    """Plot the clusters."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)

    texts = []
    fig, ax = plt.subplots(figsize=figsize)
    if labels is None:
        for (x, y), c in zip(principal_components, clusters):
            ax.scatter(x, y, c=f"C{c+1}")
    else:
        for (x, y), c, label in zip(principal_components, clusters, labels):
            ax.scatter(x, y, label=label, c=f"C{c+1}")
            texts.append(plt.text(x, y, label, fontsize=9))
        adjust_text(texts)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return fig


if __name__ == "__main__":
    _main()
