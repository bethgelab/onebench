import json
import os
import pandas as pd
import argparse
from utils import *

numeric = ["bingo", "crossmodal", "flickr30k", "coco", "mementos"]
ignore = ["image2webpage_css", "mm_safety_bench"]


# Function to load and parse a single JSON file
def parse_json_log(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Function to create a DataFrame from the parsed log files
def create_evaluation_dataframe_binary(dataset, log_files):
    results = {}
    sample_ids = set()

    for i, file_path in enumerate(log_files):

        model_name = os.path.splitext(os.path.basename(file_path))[0]
        data = parse_json_log(file_path)

        results[model_name] = {}

        for entry in data:
            qem_exists = any(
                stat["name"]["name"] == "quasi_exact_match" for stat in entry["stats"]
            )
            if not qem_exists:
                print(entry)
                continue
            id = entry["instance_id"].replace("id", "")
            key = f"{dataset}_{id}"
            sample_ids.add(key)
            target_index = [
                i
                for i, item in enumerate(entry["stats"])
                if item["name"]["name"] == "quasi_exact_match"
            ][0]
            value = entry["stats"][target_index]["mean"]
            results[model_name][key] = value

    sample_ids = sorted(sample_ids)

    df = pd.DataFrame(columns=sample_ids)

    for model_name, model_results in results.items():
        row = [model_results.get(sample_id, None) for sample_id in sample_ids]
        df.loc[model_name] = row

    return df


def create_evaluation_dataframe_numeric(dataset, log_files, metric="bleu"):
    results = {}
    sample_ids = set()

    for i, file_path in enumerate(log_files):

        model_name = os.path.splitext(os.path.basename(file_path))[0]
        data = parse_json_log(file_path)

        results[model_name] = {}

        for entry in data:
            metric_exists = any(
                stat["name"]["name"] == metric for stat in entry["stats"]
            )
            if not metric_exists:
                # print(entry)
                continue
            id = entry["instance_id"].replace("id", "")
            key = f"{dataset}_{id}"
            sample_ids.add(key)
            target_index = [
                i
                for i, item in enumerate(entry["stats"])
                if item["name"]["name"] == metric
            ][0]
            if "mementos" in dataset:
                if entry["stats"][target_index]["count"] == 0:
                    value = None
                else:
                    value = entry["stats"][target_index]["mean"] / 10.0
            else:
                value = entry["stats"][target_index]["mean"]
            results[model_name][key] = value

    sample_ids = sorted(sample_ids)

    df = pd.DataFrame(columns=sample_ids)

    for model_name, model_results in results.items():
        row = [model_results.get(sample_id, None) for sample_id in sample_ids]
        df.loc[model_name] = row

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Creating a matrix of binary metrics for the VLM dataset"
    )

    parser.add_argument(
        "--source_dir",
        default="/Users/heikekoenig/irp/lifelong_analysis/data/vhelm/score",
        help="root directory to load results",
        type=str,
    )

    parser.add_argument(
        "--des_dir",
        default="/Users/heikekoenig/irp/lifelong_analysis/data/vlm/vhelm/",
        help="root directory to save all results",
        type=str,
    )
    parser.add_argument("--metric", default="numeric", help="binary/numeric", type=str)

    args = parser.parse_args()
    remove_ds_store(args.source_dir)

    count = 0

    for dataset in sorted(os.listdir(args.source_dir)):

        # if any(substring in dataset for substring in ignore) or not any(substring in dataset for substring in numeric):
        if args.metric == "numeric":
            condition = any(substring in dataset for substring in ignore) or not any(
                substring in dataset for substring in numeric
            )
        else:
            condition = any(substring in dataset for substring in ignore + numeric)

        if condition:
            print(dataset + " ignored")
            continue

        path_src = f"{args.source_dir}/{dataset}/"
        remove_ds_store(path_src)
        files = os.listdir(path_src)
        file_paths = [os.path.join(path_src, sample_id) for sample_id in files]

        if args.metric == "numeric" and dataset != "mscoco_categorization":
            print(dataset)
            path_des = f"{args.des_dir}/{args.metric}/{dataset}/"
            if not os.path.exists(path_des):
                os.makedirs(path_des)

            if "mementos" in dataset:
                if dataset == "mementos":
                    df = create_evaluation_dataframe_numeric(
                        dataset, file_paths, "originality_gpt4v"
                    )
                else:
                    df = create_evaluation_dataframe_numeric(
                        dataset, file_paths, "prometheus_vision"
                    )

                count = count + len(df.columns)
                print(len(df.columns))

                parquet_file = os.path.join(path_des, f"originality_gpt4.parquet")
                df.to_parquet(parquet_file)
                print("Originality Scores")
                print(df.mean(axis=1))
            else:
                # df_bleu = create_evaluation_dataframe_numeric(dataset,file_paths,'bleu_1')
                # parquet_file = os.path.join(path_des, f'bleu.parquet')
                # df_bleu.to_parquet(parquet_file)
                # print("BLEU Scores")
                # print(df_bleu.mean(axis=1))

                # CIDER IS ALL ZERO

                df_rouge = create_evaluation_dataframe_numeric(
                    dataset, file_paths, "rouge_l"
                )
                count = count + len(df_rouge.columns)
                print(len(df_rouge.columns))

                parquet_file = os.path.join(path_des, f"rouge.parquet")
                df_rouge.to_parquet(parquet_file)
                print("ROUGE Scores")
                print(df_rouge.mean(axis=1))

        elif args.metric == "binary" or dataset == "mscoco_categorization":
            print(dataset)
            path_des = f"{args.des_dir}/{args.metric}/"
            if not os.path.exists(path_des):
                os.makedirs(path_des)

            df = create_evaluation_dataframe_binary(dataset, file_paths)

            count = count + len(df.columns)
            print(len(df.columns))
            parquet_file = os.path.join(path_des, f"{dataset}.parquet")
            df.to_parquet(parquet_file)
            print(df.mean(axis=1))

            # df.reset_index(inplace=True)
            # df.rename(columns={'index': 'model'}, inplace=True)
    print("Count is ", count)


if __name__ == "__main__":
    main()
