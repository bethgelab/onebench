# import torch
from PIL import Image
# from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel, AutoTokenizer
from datasets import load_dataset
import requests
import argparse
import os
os.environ["HF_HOME"] = "/mnt/qb/work/bethge/bkr536/.cache/huggingface/"
os.environ["HF_HUB_CACHE"] = "/mnt/qb/work/bethge/bkr536/.cache/huggingface/"
import re
import json
from collections import defaultdict
from utils_embed import *

def main():
    parser = argparse.ArgumentParser(description="Embedding generation with SigLip")

    parser.add_argument(
        "--source_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/lmms_eval/",
        help="root directory to load results", type=str)

    parser.add_argument(
        "--parquet_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/vlm/lmms-eval/",
        help="root directory to load results", type=str)

    parser.add_argument(
        "--result_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/vlm/siglip/lmms-eval/",
        help="root directory for saved results", type=str)

    parser.add_argument(
        "--img2id_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/vlm/id2img/",
        help="root directory for saved results", type=str)

    args = parser.parse_args()


    #Create the dictionary for binary and numeric datasets
    binary = os.listdir(f'{args.parquet_dir}/binary/')
    binary = [file_name for file_name in binary if file_name != '.DS_Store']
    binary = [re.sub(r'\.parquet$', '', file_name) for file_name in binary]

    numeric = os.listdir(f'{args.parquet_dir}/numeric/')
    numeric = [file_name for file_name in numeric if file_name != '.DS_Store']

    benchmarks = {'binary': binary, 'numeric': numeric}

    # unique_images = {}
    # image_to_rows = defaultdict(list)

    # img2id = {}
    # Load the dataset jsons
    for benchmark in benchmarks.keys():
        for dataset in benchmarks[benchmark]:
            print(dataset)
            if 'seedbench' not in dataset and 'vqav2' not in dataset:
                continue
            print(f"Processing {dataset} in {benchmark} benchmark")

            if os.path.exists(f"{args.img2id_dir}/{dataset}.json"):
                # with open(f"{args.img2id_dir}/lmms_eval_all.json") as f:
                #     img2id = json.load(f)
                continue
            else:
                print(f"Creating id2img dictionary for {dataset}")
                dataset_path = f"{args.source_dir}/{dataset}/llava_7b.json"
                if not os.path.exists(dataset_path):
                    dataset_path = f"{args.source_dir}/{dataset}/llava_v1.6_34b.json"
                with open(dataset_path) as f:
                    data = json.load(f)

                hf_path = data['model_configs']['dataset_path']
                split = data['model_configs']['test_split']

                # Load the dataset
                if "dataset_name" in data['model_configs'].keys():
                    df_dataset = load_dataset(hf_path, data['model_configs']['dataset_name'], split=split,
                                              cache_dir="/mnt/qb/work/bethge/bkr536/.cache/huggingface/")
                else:
                    df_dataset = load_dataset(hf_path, split=split,
                                              cache_dir="/mnt/qb/work/bethge/bkr536/.cache/huggingface/")

                img2id = {}
                unique_images = []
                image_to_rows = defaultdict(list)
                print(len(df_dataset))

                # Iterate over the dataset and process each image
                for i, sample in enumerate(df_dataset):
                    if dataset == 'cmmmu_val':
                        image = sample['image_1_filename']
                    elif 'coco' in dataset:
                        image = sample['file_name']
                    elif dataset == 'docvqa_val':
                        image = sample['docId']
                    elif dataset == 'flickr30k_test':
                        image = sample['filename']
                    elif dataset == 'gqa':
                        image = sample['imageId']
                    elif 'iconqa' in dataset:
                        image = sample['question_id']
                    elif dataset == 'infovqa_val':
                        image = sample['image_url']
                    elif dataset == 'mmmu_val':
                        image = sample['image_1']
                    elif dataset == 'multidocvqa_val':
                        image = sample['doc_id']
                    elif 'seedbench' in dataset:
                        if isinstance(sample['data_id'], list):
                            image = sample['data_id'][0]
                        else:
                            image = sample['data_id']
                    elif 'textcaps' in dataset:
                        image = sample['image_name']
                    elif dataset == 'vqav2_val':
                        image = sample['image_id']
                    else:
                        image = sample['image']

                    if not isinstance(image, Image.Image):
                        print(image)
                        # id = image.replace('images/','').replace('.jpg','')

                        if image not in unique_images:
                            unique_images.append(image)
                        image_to_rows[image].append(i)
                    else:
                        print(image)
                        image_hash = compute_image_hash(image)

                        if image_hash not in unique_images:
                            unique_images.append(image_hash)

                        image_to_rows[image_hash].append(i)
                # print(image_to_rows)
                for item in data['logs']:
                    doc_id = item['doc_id']
                    for image_hash, rows in image_to_rows.items():
                        if doc_id in rows:
                            img2id[f"{dataset}_{doc_id}"] = rows[0]
                            break
                with open(f"{args.img2id_dir}/{dataset}.json", "w") as f:
                    json.dump(img2id, f, indent=4)


if __name__ == "__main__":
    main()
