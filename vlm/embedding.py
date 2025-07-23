import torch
from transformers import AutoProcessor, SiglipModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import argparse
import re
import json
from utils_embed import *

root_dir = '/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/'
cache_dir = "/p/scratch/ccstdl/ghosh4_juwelsbooster/.cache"
parser = argparse.ArgumentParser(description="Embedding generation with SigLip")

parser.add_argument(
    "--source_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/lmms_eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--parquet_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/vlm/lmms-eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--result_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/results/siglip/lmms-eval/",
    help="root directory for saved results", type=str)

parser.add_argument(
    "--img2id_path", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/id2img_comb.json",
    help="root directory for saved results", type=str)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load model, processor, and tokenizer
model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir = cache_dir).to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)

#Create the dictionary for binary and numeric datasets
binary = os.listdir(f'{args.parquet_dir}/binary/')
binary = [file_name for file_name in binary if file_name != '.DS_Store']
binary = [re.sub(r'\.parquet$', '', file_name) for file_name in binary]

numeric = os.listdir(f'{args.parquet_dir}/numeric/')
numeric = [file_name for file_name in numeric if file_name != '.DS_Store']
numeric = [re.sub(r'\.parquet$', '', file_name) for file_name in numeric]

benchmarks = {'binary': binary, 'numeric': numeric}

image_embeddings = []

with open(args.img2id_path) as f:
    column_to_index = json.load(f)

file = open(f"{root_dir}data/vlm/lmms-eval/selected_numeric_features.txt", "r").readlines()
file = [f.rstrip('\n') for f in file]


embedding_to_column = {}
index = 0
for benchmark in benchmarks:
    for dataset in benchmarks[benchmark]:

        vqav2_id = []

        if dataset == 'gqa':
            continue

        if dataset == 'seedbench':
            dataset_dict = {k: v for k, v in column_to_index.items() if k.startswith(dataset) and '-2' not in k}
        else:
            dataset_dict = {k: v for k, v in column_to_index.items() if k.startswith(dataset)}

        print(f"Creating embeddings for {dataset}")
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
                                      cache_dir=cache_dir)
        else:
            df_dataset = load_dataset(hf_path, split=split, cache_dir=cache_dir)

        for column, idx in dataset_dict.items():
            if column not in file and benchmark == 'numeric':
                continue

            if dataset == 'cmmmu_val' or dataset == 'mmmu_val' or dataset == 'multidocvqa_val':
                image = df_dataset[idx]['image_1'].convert("RGB")
            elif 'iconqa' in dataset:
                image = df_dataset[idx]['query_image'].convert("RGB")
            elif dataset == 'mathvista_testmini':
                image = df_dataset[idx]['decoded_image'].convert("RGB")
            elif 'seedbench' in dataset:
                image = df_dataset[idx]['image'][0].convert("RGB")
            elif 'vqav2' in dataset:
                image_id = df_dataset[idx]['image_id']
                if image_id in vqav2_id:
                    print(f"Skipping {column}")
                    continue
                else:
                    image = df_dataset[idx]['image'].convert("RGB")
                    vqav2_id.append(image_id)

            else:
                row = df_dataset[idx]['image']
                if row is None:
                    continue
                else:
                    image = row.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
            image_embeddings.append(siglip_features)
            embedding_to_column[index] = column
            print(f"{column} - {index}")
            index += 1

image_embeddings = np.concatenate(image_embeddings, axis=0)
with open(f"{args.result_dir}index2col.json", 'w') as f:
    json.dump(embedding_to_column, f, indent=4)
np.save(f"{args.result_dir}image_embeddings.npy", image_embeddings)