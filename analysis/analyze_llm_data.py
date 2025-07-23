import json
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import scienceplots  # noqa: F401
import spacy
from analysis.elo import compute_mle, compute_tau
from analysis.io import (
    format_data_instance,
    load_long_results,
    load_pairwise_results,
    load_embeddings,
)
from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
plt.style.use(["science", "vibrant", "grid"])
plt.rcParams["font.family"] = "Times"


def main():
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        help="Path to the data file",
        default=root / "data/llm/all/instances.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=root / "data/llm/all",
    )
    args = parser.parse_args()
    analyze_llm_data(args)


def analyze_llm_data(args):
    """Embed the data instances, cluster, and visualize them."""
    with open(args.data, "r") as file:
        data = json.load(file)
    instances = [format_data_instance(instance) for instance in data.values()]
    datasets = [instance["metadata"]["source"] for instance in data.values()]
    uuids = list(data.keys())

    results_long = load_long_results(benchmark="all")
    results_pairwise = load_pairwise_results(benchmark="all")

    ranking = compute_accuracy_ranking(results_long)
    embeddings = load_embeddings(benchmark="all")

    print("Running topic modelling...")
    lda_model = run_lda(instances)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_projections = tsne.fit_transform(embeddings)

    print("Clustering the data...")
    clusterer = KMeans(n_clusters=30, random_state=42)
    clusters = clusterer.fit_predict(embeddings)
    taus = compute_taus_per_cluster(clusters, uuids, ranking, results_pairwise)
    print(taus)

    fig_datasets = plot_projection(tsne_projections, datasets)
    fig_clusters = plot_projection(tsne_projections, clusters)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_topics(args.out_dir / "topics.txt", lda_model)
    fig_datasets.savefig(args.out_dir / "datasets.pdf", bbox_inches="tight")
    fig_clusters.savefig(args.out_dir / "clusters.pdf", bbox_inches="tight")


def preprocess(nlp, text):
    """Remove stop words, tokenize, and lemmatize."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens


def compute_accuracy_ranking(df):
    """Rank the models by their average accuracy."""
    return (
        df.group_by("model")
        .agg(pl.mean("score").alias("mean_score"))
        .sort("mean_score", descending=True)
    )


def run_lda(data):
    """Run topic modelling on the data."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    processed_docs = [preprocess(nlp, doc) for doc in data]

    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    num_topics = 50
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model


def compute_taus_per_cluster(clusters, uuids, ranking, results_pairwise):
    """Calculate the tau values for each cluster."""
    taus = []
    unique_clusters = set(clusters)
    for cluster in tqdm(unique_clusters, desc="Computing taus for clusters:"):
        uuids_subset = [uuids[i] for i, c in enumerate(clusters) if c == cluster]
        subset = results_pairwise.filter(pl.col("data_instance").is_in(uuids_subset))
        ranking_subset = compute_mle(subset)
        taus.append(compute_tau(ranking["model"], ranking_subset.index))
    return taus


def plot_projection(data, labels):
    """Plot a projection of the data embeddings."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for label in set(labels):
        indices = [i for i, s in enumerate(labels) if s == label]
        ax.scatter(data[indices, 0], data[indices, 1], label=label, s=5)

    ax.legend(loc="lower right")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("t-SNE Projection of data instance embeddings")

    return fig


def save_topics(path, lda_model):
    """Save the topics to a file."""
    num_topics = lda_model.num_topics
    with open(path, "w") as file:
        for idx in range(num_topics):
            topic = lda_model.print_topic(idx)
            file.write(f"Topic {idx + 1}: {topic}\n")


if __name__ == "__main__":
    main()
