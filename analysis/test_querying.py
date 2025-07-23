import json
from argparse import ArgumentParser
from pathlib import Path

import scienceplots  # noqa: F401
from omegaconf import DictConfig, OmegaConf

from analysis import (
    compute_ranking_for_subset_dense,
    compute_tau,
)
from analysis import (
    format_data_instance,
    load_instances,
    load_or_compute_global_ranking,
    load_pairwise_results,
)
from query import query

ROOT = Path(__file__).parents[2]


def main():
    parser = ArgumentParser()
    parser.add_argument("--fig_width", type=float, default=8)
    parser.add_argument("--fig_height", type=float, default=6)
    parser.add_argument("--benchmark", type=str, default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    test_querying(args)


def test_querying(args):
    instances = load_instances(args.benchmark)
    global_ranking = load_or_compute_global_ranking(args.benchmark)
    pairwise_results = load_pairwise_results(args.benchmark)

    with open(ROOT / "hellbench/analysis/queries.txt") as f:
        queries = f.read().splitlines()

    out_path = ROOT / f"data/llm/{args.benchmark}/queries"
    out_path.mkdir(parents=True, exist_ok=True)
    for q in queries:
        config = DictConfig(
            {
                "query": q,
                "answer_query": None,
                "benchmark": args.benchmark,
                "top_k": args.k,
                "threshold": args.threshold,
                "filters": {},
                "return_similarity": True,
            }
        )
        config = OmegaConf.create(config)

        similarities, uuids = query(config)
        print(f"Found {len(uuids)} matching instances for query: {q}")
        ranking = compute_ranking_for_subset_dense(pairwise_results, uuids)
        tau = compute_tau(ranking.index, global_ranking.index)

        with open(out_path / f"{q.replace(' ', '_')}_ranking.txt", "w") as f:
            f.write(f"Query: {q}\n")
            f.write(f"Tau: {tau:.2f}\n")
            f.write(ranking.head(5).to_string())

        instances_subset = {
            f"{i}_{instances[uuid]['metadata']['source']}": {
                "text": format_data_instance(instances[uuid], json=True),
                "similarity": f"{similarity:.2f}",
            }
            for i, (uuid, similarity) in enumerate(zip(uuids, similarities))
        }
        with open(out_path / f"{q.replace(' ', '_')}_instances.json", "w") as f:
            json.dump(instances_subset, f, indent=4)


if __name__ == "__main__":
    main()
