import argparse
import json
import uuid
from pathlib import Path

import datasets
import polars as pl
import requests
from datasets.exceptions import DatasetGenerationError, DatasetNotFoundError
from jsonschema import validate
from rapidfuzz import fuzz, process

from hellbench.llm.benchmarks import ARC, MMLU, Winogrande, Hellaswag, GSM8K, TruthfulQA
from tqdm import tqdm


def _main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=root / "data/llm/leaderboard",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
        help="How many models to download",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Whether to refresh the list of model names",
    )
    args = parser.parse_args()
    download_llm_leaderboard(args)


def download_llm_leaderboard(args):
    """Download the latest leaderboard data from the Open LLM Leaderboard."""
    model_names = get_model_names(args.refresh)
    reference = "open-llm-leaderboard/details_meta-llama__Meta-Llama-3-70B-Instruct"
    benchmarks = get_benchmarks(reference)

    models = []
    dataframes = []
    while model_names and len(models) < args.num_models:
        model = model_names.pop(0)
        model_data = download_model_data(model, required_benchmarks=benchmarks)
        if model_data is not None:
            models.append(model)
            dataframes.append(model_data)
    # one dataframe per benchmark, rows are models, columns are data instances
    dataframes = [pl.concat(dfs, how="vertical") for dfs in zip(*dataframes)]

    # drop overlapping columns
    all_columns = [c for df in dataframes for c in df.columns]
    overlapping_columns = set([c for c in all_columns if all_columns.count(c) > 1])
    dataframes = [df.drop(overlapping_columns) for df in dataframes]

    # rename columns to match data instances in the original benchmarks
    dataframes = [
        match_with_benchmark(df, benchmark)
        for df, benchmark in zip(dataframes, benchmarks)
    ]
    df = pl.concat(dataframes, how="horizontal")
    df = df.with_columns(pl.Series("model", models))
    df, instances = replace_column_names_with_uuids(df)

    df = df.cast({pl.Boolean: pl.Int16, pl.Float64: pl.Int16})
    validate(instances, load_data_schema())

    args.out_dir.mkdir(exist_ok=True, parents=True)
    df.write_parquet(args.out_dir / "results.parquet")
    with open(args.out_dir / "instances.json", "w") as f:
        json.dump(instances, f, indent=4, sort_keys=True)


def get_model_names(refresh=False):
    """Return the names of all models currently on the Open LLM Leaderboard."""
    model_infos = load_model_infos(refresh)
    models = {
        name: info
        for name, info in model_infos.items()
        if "merge" not in info.get("tags", []) and info["still_on_hub"]
    }
    models = sorted(models.items(), key=lambda x: x[1].get("likes", 0), reverse=True)
    return [model[0] for model in models]


def load_model_infos(refresh=False):
    """Load or download the model infos file."""
    repo_root = Path(__file__).parent.parent
    path = repo_root / "data/llm/model_infos.json"
    if path.exists() and not refresh:
        with open(path) as f:
            model_infos = json.load(f)
    else:
        url = "https://huggingface.co/datasets/open-llm-leaderboard/dynamic_model_information/raw/main/model_infos.json"
        response = requests.get(url)
        response.raise_for_status()
        model_infos = response.json()
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(model_infos, f, indent=4, sort_keys=True)
    return model_infos


def get_benchmarks(dataset_name):
    """Get a list of available benchmarks for a given dataset."""
    data = datasets.load_dataset(
        dataset_name,
        name="results",
        split="latest",
    )
    benchmarks = data["results"][0].keys()
    benchmarks = [
        benchmark.replace("|", "_").replace(":", "_").replace("-", "_")
        for benchmark in benchmarks
        if benchmark != "all"
    ]
    return sorted(benchmarks)


def download_model_data(model_name, required_benchmarks=None):
    org, model = model_name.split("/")
    dataset_name = f"open-llm-leaderboard/details_{org}__{model}"
    print(f"Downloading available benchmarks for {model_name}")
    try:
        benchmarks = get_benchmarks(dataset_name)
    except (DatasetNotFoundError, DatasetGenerationError, ValueError):
        return None
    if benchmarks is None or not set(benchmarks).issuperset(required_benchmarks):
        return None
    model_dataframes = []
    download_config = datasets.DownloadConfig(max_retries=10)
    for benchmark in required_benchmarks:
        print(f"Downloading results for {model_name} ({benchmark})")
        data = datasets.load_dataset(
            dataset_name,
            name=benchmark,
            split="latest",
            download_config=download_config,
        )
        row = build_row(data)
        row = row.select(sorted(row.columns))
        model_dataframes.append(row)
    return model_dataframes


def build_row(data):
    """Build a single row for the leaderboard data."""
    df = data.to_polars()
    if "acc" not in df.columns and "mc1" not in df.columns:
        df = df.unnest("metrics")
    metric = "acc" if "acc" in df.columns else "mc2"
    df = df.select(["example", metric]).unique("example")
    return df.transpose(column_names="example")


def match_with_benchmark(df, benchmark):
    print(f"Matching dataset for {benchmark}")
    subset = "_".join(benchmark.split("_")[2:-1])
    benchmark = benchmark.split("_")[1]
    benchmark_to_class = {
        "arc": ARC,
        "winogrande": Winogrande,
        "hendrycksTest": MMLU,
        "hellaswag": Hellaswag,
        "gsm8k": GSM8K,
        "truthfulqa": TruthfulQA,
    }
    if benchmark == "hendrycksTest":
        benchmark = benchmark_to_class[benchmark](subset)
    else:
        benchmark = benchmark_to_class[benchmark]()
    data_instances = benchmark.get_data_instances()
    questions = benchmark.get_benchmark_questions()
    mapping = {
        old_name: json.dumps(data_instances[find_match_index(old_name, questions)])
        for old_name in tqdm(df.columns)
    }
    return df.rename(mapping)


def find_match_index(old_name, questions):
    """Find the matching data instance."""
    match, score, match_index = process.extractOne(
        old_name, questions, scorer=fuzz.WRatio, score_cutoff=50
    )
    assert (
        score > 90
    ), f"No match found for >>{old_name}<<. Closest match: {match} with score {score}"
    return match_index


def replace_column_names_with_uuids(df):
    """Replace instance IDs with UUIDs."""
    instances = {}
    for column in df.columns:
        if column == "model":
            continue
        hex = uuid.uuid4().hex
        df = df.rename({column: hex})
        instances[hex] = json.loads(column)
    return df, instances


def load_data_schema():
    """Load the JSON schema for the data instances."""
    repo_root = Path(__file__).parent.parent
    schema_path = repo_root / "data/llm/data_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    return schema


if __name__ == "__main__":
    _main()
