"""Download per-instance results from HELM."""

import argparse
import subprocess
from pathlib import Path

from google.cloud import storage


def _main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Where to save the data",
        default=root / "data/llm/helm/download",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files",
    )
    args = parser.parse_args()
    download_helm(args)


def download_helm(args):
    """Download per-instance results from HELM."""
    client = storage.Client()
    bucket = client.bucket("crfm-helm-public")
    blobs = bucket.list_blobs(prefix="lite/benchmark_output/runs")
    blobs = [blob for blob in blobs if is_relevant(blob)]
    args.out_dir.mkdir(exist_ok=True, parents=True)
    for blob in blobs:
        maybe_download_blob(blob, args.out_dir, args.overwrite)


def is_relevant(blob):
    """Check if a blob is relevant for download."""
    names = ["run_spec.json", "instances.json", "display_predictions.json"]
    scenarios = [
        "commonsense",
        "gsm",
        "legalbench",
        "math",
        "med_qa",
        "mmlu",
        "narrative_qa",
        "natural_qa",
        "wmt_14",
    ]
    return Path(blob.name).name in names and any(s in blob.name for s in scenarios)


def maybe_download_blob(blob, out_dir, overwrite=False):
    """Download a blob. Skip if it already exists and overwrite is False."""
    blob_path = Path(blob.name)
    relative_path = get_relative_path(blob_path)
    absolute_path = out_dir / relative_path
    if absolute_path.exists() and not overwrite:
        print(f"Skipping {blob.name} as {relative_path} already exists")
        return
    print(f"Downloading {blob_path} to {relative_path}")
    subprocess.run(
        ["gsutil", "cp", f"gs://crfm-helm-public/{blob.name}", absolute_path],
        stderr=subprocess.DEVNULL,
    )


def get_relative_path(blob_path):
    """Return local download path relative to the output directory."""
    parameters = extract_parameters(blob_path)
    scenario = parameters["scenario"]
    subscenario = extract_subscenario(parameters)
    if subscenario is not None:
        scenario = f"{scenario}_{subscenario}"
    return Path(scenario) / parameters["model"] / blob_path.name


def extract_parameters(blob_path):
    """Extract run parameters from the blob path."""
    scenario, rest = blob_path.parent.name.split(":")
    parameters = dict(p.split("=") for p in rest.split(","))
    parameters["scenario"] = scenario
    return parameters


def extract_subscenario(parameters):
    """Extract subscenario name from the parameters if it exists."""
    subscenario_keys = ["dataset", "subset", "subject", "mode", "language_pair"]
    key = next((k for k in subscenario_keys if k in parameters), None)
    return parameters.get(key)


if __name__ == "__main__":
    _main()
