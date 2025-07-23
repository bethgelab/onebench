import json
import os
import choix
from datasets import load_dataset
from utils import *
from utils_embed import extract_dataset_name
import pandas as pd

def return_retrievals(indices, args, index_to_col, a_matrix, text_prompt, col_to_ds, emb_type='img'):
    rank = []
    for i, elem in enumerate(indices):
        col = index_to_col[str(elem)]
        print(col)
        dataset = extract_dataset_name(col)
        print(dataset)

        if col in a_matrix.columns:
            rank.append(col)
        else:
            continue

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
                                      cache_dir=args.cache_dir)
        else:
            df_dataset = load_dataset(hf_path, split=split, cache_dir=args.cache_dir)

        save_dir = f"{args.result_dir}/results/{args.query_images}_{emb_type}/{text_prompt.replace(' ', '_')}"
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

    rank.append('model')
    print(rank)
    print()
    ranks = a_matrix.select(rank)
    print(ranks)

    long_ranks = load_long(ranks)
    pairwise_ranks = to_pairwise(long_ranks, long=True)

    pair_list, models = compute_pairwise(pairwise_ranks)
    lsr_params = choix.lsr_pairwise(len(models), pair_list, alpha=0.05)
    lsr_rank = pd.Series(lsr_params, index=models).sort_values(ascending=False)

    print(lsr_rank)

def return_retrievals2(indices, args, index_to_col, text_prompt, col_to_ds, emb_type='img'):
    rank = []
    for i, elem in enumerate(indices):
        col = index_to_col[str(elem)]
        print(col)
        dataset = extract_dataset_name(col)


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
                                      cache_dir=args.cache_dir)
        else:
            df_dataset = load_dataset(hf_path, split=split, cache_dir=args.cache_dir)

        save_dir = f"{args.result_dir}/results/{args.query_images}_{emb_type}/{text_prompt.replace(' ', '_')}"
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
