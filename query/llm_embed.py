import argparse
import json
import warnings
from pathlib import Path
from itertools import zip_longest

import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*"
)

BASE_DIR = Path(__file__).parents[2] / "data/llm"


def _main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, help="Benchmark to embed", default="all"
    )
    args = parser.parse_args()

    benchmark_path = BASE_DIR / args.benchmark
    question_embeddings_path = benchmark_path / "question_embeddings.npy"
    answer_embeddings_path = benchmark_path / "answer_embeddings.npy"
    instances_path = benchmark_path / "instances.json"

    with open(instances_path, "r") as file:
        instances = json.load(file)

    questions = [extract_question(instance) for instance in instances.values()]
    answers = [extract_answer(instance) for instance in instances.values()]

    question_embeddings = embed(questions)
    answer_embeddings = embed(answers)

    np.save(question_embeddings_path, question_embeddings)
    np.save(answer_embeddings_path, answer_embeddings)


def extract_question(instance):
    """Extract the question from the data instance."""
    return instance["question"]["text"]


def extract_answer(sample, labels=None):
    """Extract the answer from the data instance."""
    answers = [answer["text"] for answer in sample["references"]]
    if labels is not None:
        answers = [
            f"{label}. {reference}" for label, reference in zip_longest(labels, answers)
        ]
    return "\n".join(answers)


def embed(instances):
    """Embed the data instances."""
    model = SentenceTransformer("all-MiniLM-L12-v2")
    if len(instances) > 1000:
        print("Embedding data instances...")
        return model.encode(instances)
    else:
        return model.encode(instances, show_progress_bar=False)


if __name__ == "__main__":
    _main()
