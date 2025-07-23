"""Parse the HELM data."""

import argparse
import json
import uuid
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process
from tqdm import tqdm

from hellbench.llm.benchmarks import (
    GSM8K,
    LegalBench,
    NarrativeQA,
    NaturalQA,
    OpenBookQA,
    Math,
    MedQA,
    MMLU,
    WMT14,
)


def _main():
    root = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to the downloaded HELM data",
        default=root / "data/llm/helm/download",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the result",
        default=root / "data/llm/helm",
    )
    args = parser.parse_args()
    parse_helm(args)


def parse_helm(args):
    """Parse the downloaded HELM data and save it as a binary NumPy array."""
    models = set()
    excluded = set()
    dataframes = []

    for benchmark_dir in sorted(args.data_dir.iterdir()):
        benchmark_df = pl.DataFrame()
        with open(next(benchmark_dir.iterdir()) / "instances.json") as f:
            instance_id_to_text = {
                instance["id"]: instance["input"]["text"] for instance in json.load(f)
            }
        for model_dir in sorted(benchmark_dir.iterdir()):
            row = build_row(model_dir, instance_id_to_text)
            average = row.drop_nulls().mean_horizontal().item()
            number_of_values = row.drop_nulls().shape[1]
            print(f"Model: {model_dir.name}, Benchmark: {benchmark_dir.name}")
            print(f"Average score: {average} from {number_of_values} values")
            benchmark_df = concatenate_row_to_dataframe(benchmark_df, row)
            models.add(model_dir.name)

        # one dataframe per benchmark, rows are models, columns are data instances
        dataframes.append(benchmark_df)

    print("Accepted models:", len(models))
    print("Excluded models:", len(excluded))

    # rename columns to match data instances in the original benchmarks
    benchmarks = [
        benchmark_dir.name for benchmark_dir in sorted(args.data_dir.iterdir())
    ]
    dataframes = [
        match_with_benchmark(df, benchmark)
        for df, benchmark in zip(dataframes, benchmarks)
    ]
    df = pl.concat(dataframes, how="horizontal")
    df = df.with_columns(pl.Series("model", list(sorted(models))))
    df, instances = replace_column_names_with_uuids(df)

    args.out_dir.mkdir(exist_ok=True, parents=True)
    df.write_parquet(args.out_dir / "results.parquet")
    with open(args.out_dir / "instances.json", "w") as f:
        json.dump(instances, f, indent=4, sort_keys=True)


def concatenate_row_to_dataframe(df, row):
    """Concatenate a row to a dataframe. Handle missing columns."""
    if df.shape[1] == 0:
        return row

    new_columns = set(row.columns) - set(df.columns)

    for col in new_columns:
        df = df.with_columns(pl.lit(None).cast(row[col].dtype).alias(col))

    missing_columns = set(df.columns) - set(row.columns)

    for col in missing_columns:
        row = row.with_columns(pl.lit(None).cast(df.schema[col]).alias(col))

    row = row.select(df.columns)

    df = df.vstack(row)

    return df


def load_instances(benchmark_dir):
    """Load the scenario instances."""
    model_dir = next(benchmark_dir.iterdir())
    with open(model_dir / "instances.json") as f:
        data = {instance["id"]: instance["input"]["text"] for instance in json.load(f)}
    return data


def build_row(model_dir, instance_id_to_text):
    """Aggregate the predictions for a single model into a row."""
    scenario = model_dir.parent.name
    metric = get_default_metric(scenario)
    display_predictions = json.load((model_dir / "display_predictions.json").open())
    data = {
        instance_id_to_text[instance["instance_id"]]: instance["stats"][metric]
        for instance in display_predictions
    }
    return pl.DataFrame(data)


def match_with_benchmark(df, benchmark):
    print(f"Matching dataset for {benchmark}")
    name_to_class = {
        "natural": NaturalQA,
        "narrative": NarrativeQA,
        "commonsense": OpenBookQA,
        "gsm": GSM8K,
        "legalbench": LegalBench,
        "math": Math,
        "med": MedQA,
        "mmlu": MMLU,
        "wmt": WMT14,
    }
    name = benchmark.split("_")[0]
    if name in ["mmlu", "legalbench"]:
        subset = "_".join(benchmark.split("_")[1:])
        benchmark = name_to_class[name](subset)
    elif name in ["wmt", "natural"]:
        subset = "_".join(benchmark.split("_")[2:])
        benchmark = name_to_class[name](subset)
    else:
        benchmark = name_to_class[name]()
    data_instances = benchmark.get_data_instances()
    questions = benchmark.get_benchmark_questions()
    mapping = {
        old_name: json.dumps(data_instances[find_match_index(old_name, questions)])
        for old_name in tqdm(df.columns)
    }
    return df.rename(mapping)


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


def find_match_index(old_name, questions):
    """Find the matching data instance."""
    try:
        match, score, match_index = process.extractOne(
            old_name, questions, scorer=fuzz.QRatio, score_cutoff=50
        )
    except TypeError:
        print(f"No match found for {old_name}")
    assert (
        score > 90
    ), f"No match found for {old_name}. Closest match: {match} with score {score}"
    return match_index


def get_default_metric(scenario):
    """Return the metric to use for a given scenario."""
    if scenario.startswith("wmt"):
        return "bleu_4"
    if scenario.startswith("math"):
        return "math_equiv_chain_of_thought"
    if scenario.startswith("gsm"):
        return "final_number_exact_match"
    if scenario.startswith("legalbench") or scenario.startswith("med_qa"):
        return "quasi_exact_match"
    if scenario.startswith("natural_qa") or scenario.startswith("narrative_qa"):
        return "f1_score"
    return "exact_match"


def load_schema():
    repo_root = Path(__file__).parent.parent
    schema_path = repo_root / "data/llm/data_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    return schema


if __name__ == "__main__":
    _main()
