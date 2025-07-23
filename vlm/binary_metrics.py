# BINARY METRIC

import os
import json
import pandas as pd
import argparse
import difflib
import string
from binary_functions import *
from utils import *

parser = argparse.ArgumentParser(description="Creating a matrix of binary metrics for the VLM dataset")

parser.add_argument(
    "--source_dir", default="/Users/heikekoenig/irp/lifelong_analysis/data/lmms_eval",
    help="root directory to load results", type=str)

parser.add_argument(
    "--des_dir", default="/Users/heikekoenig/irp/lifelong_analysis/data/vlm/lmms-eval/binary",
    help="root directory to save all results", type=str)

parser.add_argument("--dataset",
                    default='pope',
                    help="Which dataset to use. Refer to the README for more", type=str)

args = parser.parse_args()


dataset = args.dataset
path_src = f'{args.source_dir}/{dataset}/'
path_des = f'{args.des_dir}/'


if not os.path.exists(path_des):
    os.makedirs(path_des)
a_matrix = {}
remove_ds_store(path_src)

if dataset == 'mmbench_en_dev':
    ref_model1 = 'llava_7b.json'
    with open(os.path.join(path_src, ref_model1)) as f:
        ref_data1 = json.load(f)['logs']
    id_list1 = [ref_data1[i]['doc_id'] for i in range(len(ref_data1))]

    ref_model2 = 'instructblip-vicuna-7b.json'
    with open(os.path.join(path_src, ref_model2)) as f:
        ref_data2 = json.load(f)['logs']
    id_list2 = [ref_data2[i]['doc_id'] for i in range(len(ref_data2))]

    difference = [item for item in id_list1 if item not in id_list2]
    old_models = ['llava_1.6_13b.json', 'llava_1.6_mistral_7b.json','llava_1.6_vicuna_7b.json','llava_7b.json','llava_13b.json']

for model in os.listdir(path_src):
    print(model)

    with open(os.path.join(path_src, model)) as f:
        data = json.load(f)['logs']

    if 'model' not in a_matrix:
        a_matrix['model'] = []

    a_matrix['model'].append(model.replace('.json', ''))
    if dataset == 'mmbench_en_dev':
        if model not in old_models:
            for id in difference:
                print(id)
                key = f'{dataset}_{id}'
                if key not in a_matrix:
                    a_matrix[key] = []
                a_matrix[key].append(None)

    for index, entry in enumerate(data):
        id = entry['doc_id']
        key = f'{dataset}_{id}'
        if key not in a_matrix:
            a_matrix[key] = []

        if 'exact_match' in list(entry.keys()):
            value = exact_match(dataset,entry)

        elif dataset == 'chartqa':
            value = entry['relaxed_overall']

        elif 'mmmu' in dataset:
            dict_key = dataset.split('_')[0]
            pred = entry[f'{dict_key}_acc']['parsed_pred']
            if isinstance(pred, list):
                pred = next((item for item in pred if isinstance(item, str)), '')
            ratio = difflib_ratio(entry[f'{dict_key}_acc']['answer'].lower(), pred.lower().strip('.').strip(' '))
            if ratio >= 0.5:
                value = 1.0
            else:
                value = 0.0

        elif 'iconqa' in dataset:
            value = iconqa(entry)

        elif dataset == 'mathvista_testmini':
            value = 1 if entry['gpt_eval_score']['true_false'] else 0

        elif 'mmbench' in dataset:
            value = mmbench(entry)

        elif dataset == 'mme':
            value = entry[list(entry.keys())[-1]]["score"]

        elif dataset == 'mmvet':
            value = entry['gpt_eval_score']['score']

        elif dataset == 'pope':
            value = entry["pope_accuracy"]["score"]

        elif dataset == 'seedbench' or dataset == 'seedbench-2':
            if model == 'llava_1.6_mistral_7b.json' or model == 'instructblip-vicuna-7b.json' or model == 'qwen_vl_chat.json' or model == 'idefics2-8b.json':
                pred = entry["seed_all"]["pred"].lower()
            else:
                pred = entry["filtered_resps"][0].lower()
            if pred == "":
                value = None
            else:
                # pred = pred.replace(' ', '').replace('.', '')
                pred = pred.replace("the answer is ","").replace('.', '').lower()
                if pred == "":
                    value = None
                else:
                    value = entry["seed_all"]["answer"].lower() == pred

        else:
            value = entry['anls']
        a_matrix[key].append(value)

df = pd.DataFrame(a_matrix)
df.set_index(df.columns[0], inplace=True)
parquet_file = os.path.join(path_des, f'{dataset}.parquet')
df.to_parquet(parquet_file)

print(df.mean(axis = 1))