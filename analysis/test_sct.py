import argparse
import numpy as np
from tqdm import tqdm

from analysis import (
    compute_plackett_luce_pairwise_dense,
    compute_tau,
    compute_score_ranking,
)
from analysis import (load_pairwise_results,
                      load_results,
                      load_synthetic_model_parameters)

from analysis import (borda_count, dowdall_score)

def correlation(global_ranking, pairwise_results):
    results = {}

    taus_borda = []
    taus_dowdall = []
    taus_plackett = []
    for _ in range(1):

        for name, method in [
            ("borda", borda_count),
            ("dowdall", dowdall_score),
            # ("plackett-luce", compute_plackett_luce_pairwise),
        ]:
            ranking = method(pairwise_results)
            if name == "borda":
                taus_borda.append(compute_tau(global_ranking.index, ranking.index))
            elif name == "dowdall":
                taus_dowdall.append(compute_tau(global_ranking.index, ranking.index))
            elif name == "plackett-luce":
                taus_plackett.append(compute_tau(global_ranking.index, ranking.index))


    results["borda"] = [np.mean(taus_borda), np.std(taus_borda)]
    results["dowdall"] = [np.mean(taus_dowdall), np.std(taus_dowdall)]
    results["plackett-luce"] = [np.mean(taus_plackett), np.std(taus_plackett)]

    return results

def test_separability(pairwise_results):
    """Test separability of different ranking methods."""
    first_half = pairwise_results.sample(frac=0.5)
    second_half = pairwise_results.drop(first_half.index)

    results = {}
    for name, method in [
        ("borda", borda_count),
        ("dowdall", dowdall_score),
        ("plackett-luce", compute_plackett_luce_pairwise_dense),
    ]:
        ranking1 = method(first_half)
        ranking2 = method(second_half)
        ranking_full = method(pairwise_results)

        consistent_pairs = []
        models = list(ranking1.index)
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                if (
                    ranking1[model_a] > ranking1[model_b]
                    and ranking2[model_a] > ranking2[model_b]
                ):
                    consistent_pairs.append((model_a, model_b))
                elif (
                    ranking1[model_b] > ranking1[model_a]
                    and ranking2[model_b] > ranking2[model_a]
                ):
                    consistent_pairs.append((model_b, model_a))

        maintained_count = 0
        for model_a, model_b in consistent_pairs:
            if ranking_full[model_a] > ranking_full[model_b]:
                maintained_count += 1

        if consistent_pairs:
            percentage = (maintained_count / len(consistent_pairs)) * 100
        else:
            percentage = 0.0

        print(f"Method: {name}")
        print(f"Maintained {maintained_count} out of {len(consistent_pairs)} pairs")

        results[name] = percentage

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=5, help="Number of times to repeat the analysis"
    )
    parser.add_argument(
        "--test", type=str, default="correlation", help="Number of times to repeat the analysis"
    )
    args = parser.parse_args()

    # benchmarks = ["helm", "leaderboard", "vhelm", "synthetic"]
    benchmarks = ["leaderboard"]

    results = {}


    for benchmark in tqdm(benchmarks):
        # try:
        pairwise_results = load_pairwise_results(benchmark=benchmark).to_pandas()
        benchmark_results = {
            "borda": [],
            "dowdall": [],
            "plackett-luce": [],
        }

        if args.test == "separability":
            for _ in range(args.n):
                run_results = test_separability(pairwise_results)
                for method in run_results:
                    benchmark_results[method].append(run_results[method])

            results[benchmark] = {
                method: {"mean": np.mean(values), "std": np.std(values)}
                for method, values in benchmark_results.items()
            }
        elif args.test == "correlation":
            # if benchmark == "vhelm":
            #     results = load_filtered_vhelm_results()
            # else:
            results = load_results(benchmark)
            print(results)

            if benchmark == "synthetic":
                ranking = load_synthetic_model_parameters()
            else:
                ranking = compute_score_ranking(results, benchmark)
            run_results = correlation(ranking, pairwise_results)
            print(run_results)
            # for method in run_results:
            #     benchmark_results[method].append(run_results[method])
            # results[benchmark] = {
            #     method: {"mean": np.mean(values), "std": np.std(values)}
            #     for method, values in benchmark_results.items()
            # }

        # except Exception as e:
        #     print(f"Error processing benchmark {benchmark}: {e}")
        #     results[benchmark] = {
        #         method: {"mean": 0.0, "std": 0.0}
        #         for method in ["borda", "dowdall", "plackett-luce"]
        #     }

    # if args.test == "separability":
    #     print(f"\nSeparability Test Results (percentages, n={args.n}):")
    # elif args.test == "correlation":
    #     print(f"\nCorrelation Test Results (taus, n={args.n}):")
    # print("-" * 100)
    # print(f"{'Benchmark':<15} {'Borda':>20} {'Dowdall':>20} {'Plackett-Luce':>20}")
    # print(f"{'':15} {'mean ± std':>20} {'mean ± std':>20} {'mean ± std':>20}")
    # print("-" * 100)
    # for benchmark in benchmarks:
    #     res = results[benchmark]
    #     print(
    #         f"{benchmark:<15} "
    #         f"{res['borda']['mean']:>8.2f} ± {res['elo']['std']:>5.2f} "
    #         f"{res['dowdall']['mean']:>8.2f} ± {res['lmarena']['std']:>5.2f} "
    #         f"{res['plackett-luce']['mean']:>8.2f} ± {res['plackett-luce']['std']:>5.2f}"
    #     )
    # print("-" * 100)


if __name__ == "__main__":
    main()
