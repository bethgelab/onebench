import torch
import numpy as np
import json
from transformers import AutoProcessor, SiglipModel, AutoTokenizer
import argparse
import os
import re
from datasets import load_dataset
import pandas as pd
from utils_embed import *

root_dir = '/mnt/qb/work/bethge/bkr536/lifelong/'
cache_dir = "/p/scratch/ccstdl/ghosh4_juwelsbooster/.cache"
parser = argparse.ArgumentParser(description="Embedding generation with SigLip")

parser.add_argument(
    "--source_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/lmms_eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--parquet_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/data/vlm/lmms-eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--result_dir", default="/mnt/qb/work/bethge/bkr536/lifelong/results/siglip/lmms-eval/",
    help="root directory for saved results", type=str)

parser.add_argument(
    "--text", default="car",
    help="text prompt to query", type=str)

args = parser.parse_args()

with open(f'{root_dir}/results/siglip/lmms-eval/index2col_flick.json') as f:
    index_to_col = json.load(f)


with open(f'{root_dir}/data/id2img_comb.json') as f:
    col_to_ds = json.load(f)

binary = os.listdir(f'{args.parquet_dir}/binary/')
binary = [file_name for file_name in binary if file_name != '.DS_Store']
binary = [re.sub(r'\.parquet$', '', file_name) for file_name in binary]
print(binary)

numeric = os.listdir(f'{args.parquet_dir}/numeric/')
numeric = [file_name for file_name in numeric if file_name != '.DS_Store']
numeric = [re.sub(r'\.parquet$', '', file_name) for file_name in numeric]
print(numeric)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# Load the model, processor, and tokenizer
model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir = cache_dir).to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)

embeddings = np.load(f'{root_dir}results/siglip/lmms-eval/image_embeddings_flick.npy')
print(embeddings.shape)
embeddings = torch.tensor(embeddings).to(device)
embeddings = torch.nn.functional.normalize(embeddings, dim=1)  #NEW

def get_text_embedding(text_prompt):
    inputs = processor(text=[text_prompt], return_tensors="pt", padding=False, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    text_embedding = torch.nn.functional.normalize(outputs, dim=1) #NEW 
    return outputs

text_prompt = args.text.replace('_', ' ')

text_embedding = get_text_embedding(text_prompt)

dot_products = torch.matmul(embeddings, text_embedding.T).squeeze()
top10_indices = torch.argsort(dot_products, descending=True)[:10]
indices = top10_indices.cpu().numpy().tolist()

print("Top 10 indices:", indices)

a_matrix = pd.read_parquet(f'{args.parquet_dir}/results.parquet')
# a_matrix = pd.read_parquet(f'{args.parquet_dir}/numeric.parquet')

rank = []
for i, elem in enumerate(indices):
    col = index_to_col[str(elem)]
    print(col)
    dataset = extract_dataset_name(col)
    print(dataset)

    if col in a_matrix.columns:
        inst = a_matrix[col]
        print(inst)
    else:
        continue
    rank.append(inst)

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

    save_dir = f"{args.result_dir}/query/{text_prompt.replace(' ','_')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ds_entry = col_to_ds[col]
    if dataset == 'cmmmu_val' or dataset == 'mmmu_val' or dataset == 'multidocvqa_val':
        image = df_dataset[ds_entry]['image_1'].convert("RGB")
    elif 'iconqa' in dataset:
        image = df_dataset[ds_entry]['query_image'].convert("RGB")
    elif dataset == 'mathvista_testmini':
        image = df_dataset[ds_entry]['decoded_image'].convert("RGB")
    elif 'seedbench' in dataset:
        image = df_dataset[ds_entry]['image'][0].convert("RGB")
    else:
        row = df_dataset[ds_entry]['image']
        if row is None:
            continue
        else:
            image = row.convert("RGB")

    image.save(f"{save_dir}/{i}.png")
ranks = pd.concat(rank, axis=1)
print(ranks)

