import networkx as nx
from collections import defaultdict
import pandas as pd

from analysis import load_pairwise_results
from analysis import compute_plackett_luce_pairwise_dense


def count_short_cycles(benchmark_name, max_length=4):
    """
    Count cycles up to a specified maximum length.
    If no cycles exist, return the topological ordering of models.
    """
    try:
        pairwise_results = load_pairwise_results(benchmark_name)
        all_models = set(
            pairwise_results["model_a"].unique().to_list()
            + pairwise_results["model_b"].unique().to_list()
        )

        G = nx.DiGraph()
        for model in all_models:
            G.add_node(model)

        model_pairs = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0})

        for row in pairwise_results.iter_rows():
            model_a, model_b, _, winner = row
            pair_key = (model_a, model_b) if model_a < model_b else (model_b, model_a)

            if winner == "model_a":
                if model_a < model_b:
                    model_pairs[pair_key]["a_wins"] += 1
                else:
                    model_pairs[pair_key]["b_wins"] += 1
            elif winner == "model_b":
                if model_a < model_b:
                    model_pairs[pair_key]["b_wins"] += 1
                else:
                    model_pairs[pair_key]["a_wins"] += 1
            elif winner == "tie":
                model_pairs[pair_key]["ties"] += 1

        win_margins = []
        for (model_a, model_b), counts in model_pairs.items():
            a_wins = counts["a_wins"]
            b_wins = counts["b_wins"]

            if a_wins > b_wins:
                G.add_edge(model_a, model_b, weight=a_wins - b_wins)
                win_margins.append((model_a, model_b, a_wins - b_wins))
            elif b_wins > a_wins:
                G.add_edge(model_b, model_a, weight=b_wins - a_wins)
                win_margins.append((model_b, model_a, b_wins - a_wins))

        has_cycles = not nx.is_directed_acyclic_graph(G)

        if not has_cycles:
            topo_order = list(nx.topological_sort(G))
            topo_order.reverse()
            scores = {model: len(topo_order) - i for i, model in enumerate(topo_order)}

            return {
                "benchmark": benchmark_name,
                "has_cycles": False,
                "total_models": len(all_models),
                "total_pairs": len(model_pairs),
                "cycle_counts": {i: 0 for i in range(3, max_length + 1)},
                "topological_ordering": topo_order,
                "model_scores": scores,
                "graph": G,
            }

        cycle_counts = {}
        for length in range(3, max_length + 1):
            try:
                cycles = list(nx.simple_cycles(G, length_bound=length))
                cycles = [c for c in cycles if len(c) == length]
                cycle_counts[length] = len(cycles)

            except Exception as e:
                print(
                    f"Error counting cycles of length {length} in {benchmark_name}: {e}"
                )
                cycle_counts[length] = "Error"

        return {
            "benchmark": benchmark_name,
            "has_cycles": True,
            "total_models": len(all_models),
            "total_pairs": len(model_pairs),
            "cycle_counts": cycle_counts,
            "graph": G,  # Return the graph for further analysis
        }
    except Exception as e:
        print(f"Error processing benchmark {benchmark_name}: {e}")
        return {
            "benchmark": benchmark_name,
            "has_cycles": False,
            "total_models": 0,
            "total_pairs": 0,
            "error": str(e),
        }


def get_majority_ranking(benchmark_name):
    """
    Get the majority ranking for a benchmark if it has no cycles.
    Returns a pandas Series with models as index and scores as values.
    """
    result = count_short_cycles(benchmark_name)

    if not result.get("has_cycles", True) and "model_scores" in result:
        # Return as pandas Series for easy sorting
        return pd.Series(result["model_scores"]).sort_values(ascending=False)
    else:
        print(
            f"Warning: {benchmark_name} contains preference cycles. Cannot determine a unique majority ranking."
        )
        return pd.Series()


def is_valid_topological_order(G, ordering):
    """
    Check if the given ordering is a valid topological ordering of the graph.

    In a valid topological ordering, for each edge (u,v), u must come before v in the ordering.
    In our case, if model A beats model B, then A should be ranked higher (come before) B.
    """
    node_pos = {node: i for i, node in enumerate(ordering)}
    for u, v in G.edges():
        if node_pos[u] > node_pos[v]:
            return False

    return True


def compare_with_plackett_luce(benchmark_name):
    """
    Simply verify if the Plackett-Luce ranking is a valid topological ordering.
    If not, find specific violations in the ranking.
    """
    # First run count_short_cycles to get topological ordering if available
    cycle_info = count_short_cycles(benchmark_name)

    if cycle_info.get("has_cycles", True):
        print(
            f"Warning: {benchmark_name} contains preference cycles. Topological ordering is not unique."
        )
        return {"benchmark": benchmark_name, "has_cycles": True}

    pairwise_results = load_pairwise_results(benchmark_name)

    pl_ranking = compute_plackett_luce_pairwise_dense(pairwise_results)
    pl_ordering = pl_ranking.index.tolist()

    G = cycle_info.get("graph", nx.DiGraph())
    is_valid = True
    violations = []

    node_pos = {node: i for i, node in enumerate(pl_ordering)}

    for u, v in G.edges():
        if node_pos[u] > node_pos[v]:
            is_valid = False
            u_rank = node_pos[u] + 1
            v_rank = node_pos[v] + 1
            violations.append((u, v, u_rank, v_rank))

            if len(violations) >= 5:
                break

    return {
        "benchmark": benchmark_name,
        "has_cycles": False,
        "plackett_luce_is_valid_topo": is_valid,
        "violations": violations,
    }


def main():
    benchmarks = ["helm", "leaderboard", "vhelm", "lmms-eval", "synthetic"]
    results = {}
    max_length = 6  # Define max_length for counting and display

    print("Counting short cycles in benchmarks...")
    for benchmark in benchmarks:
        print(f"Processing {benchmark}...")
        results[benchmark] = count_short_cycles(benchmark, max_length=max_length)

    print(f"\nShort Cycles Summary (Length 3-{max_length}):")
    print("-" * 100)
    header = f"{'Benchmark':<15} {'Total Models':<15} {'Total Pairs':<15} "
    for length in range(3, max_length + 1):
        header += f"{'Length ' + str(length):<15} "
    print(header)
    print("-" * 100)

    for benchmark in benchmarks:
        data = results[benchmark]
        total_models = data.get("total_models", 0)
        total_pairs = data.get("total_pairs", 0)
        cycle_counts = data.get("cycle_counts", {})

        print(f"{benchmark:<15} {total_models:<15} {total_pairs:<15}", end=" ")
        for length in range(3, max_length + 1):
            count = cycle_counts.get(length, 0)
            print(f"{count:<15}", end=" ")
        print()

    print("-" * 100)

    print("\nStatistics:")
    print("-" * 70)

    for benchmark in benchmarks:
        data = results[benchmark]
        if not data.get("has_cycles", False):
            print(f"{benchmark}: No cycles detected")
            continue

        total_models = data.get("total_models", 0)
        cycle_counts = data.get("cycle_counts", {})

        if total_models >= 3 and isinstance(cycle_counts.get(3), int):
            max_triples = total_models * (total_models - 1) * (total_models - 2) // 6
            if max_triples > 0:
                triple_percentage = (cycle_counts.get(3, 0) / max_triples) * 100
                print(
                    f"{benchmark}: {cycle_counts.get(3, 0)} out of {max_triples} possible model triples form a cycle ({triple_percentage:.2f}%)"
                )

    print("-" * 70)

    print("\nVerifying if Plackett-Luce ranking is a valid topological ordering:")
    print("-" * 70)

    for benchmark in benchmarks:
        data = results[benchmark]
        if not data.get("has_cycles", True):
            print(f"\nAnalyzing {benchmark}...")
            comparison = compare_with_plackett_luce(benchmark)

            is_valid = comparison.get("plackett_luce_is_valid_topo", False)
            print(f"Plackett-Luce ranking is a valid topological ordering: {is_valid}")

            if not is_valid and comparison.get("violations"):
                print(
                    "\nExamples of violations (where majority preference disagrees with PL ranking):"
                )
                print("(model_winner, model_loser, winner_PL_rank, loser_PL_rank)")

                for winner, loser, winner_rank, loser_rank in comparison.get(
                    "violations"
                ):
                    print(
                        f"  {winner} > {loser}, but ranked {winner_rank} vs {loser_rank}"
                    )


if __name__ == "__main__":
    main()
