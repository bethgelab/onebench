# NUMERIC METRIC

import os
import json
import pandas as pd
import difflib
import nltk
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider_scorer import CiderScorer
from numeric_functions import *
from utils import *

nltk.download('wordnet')

parser = argparse.ArgumentParser(description="Creating a matrix of numeric metrics for the VLM dataset")

parser.add_argument(
    "--source_dir", default="/Users/heikekoenig/irp/lifelong_analysis/data/vhelm/score/",
    help="root directory to load jsons", type=str)
parser.add_argument(
    "--des_dir", default="/Users/heikekoenig/irp/lifelong_analysis/data/vlm/lmms-eval/numeric/",
    help="root directory to load jsons", type=str)
parser.add_argument("--dataset",
                    default='flickr30k_test',
                    help="Which dataset to use. Refer to the README for more", type=str)

parser.add_argument("--task",
                    default=1,
                    help="Which dataset to use. Refer to the README for more", type=int)

args = parser.parse_args()

id2benchmark = {0: ['coco2014_cap_val'],
                1: ['coco2017_cap_val'],
                2: ['flickr30k_test'],
                3: ['llava_in_the_wild', 'mmvet','multidocvqa_val', 'vqav2_val'],
                4: ['refcoco+_bbox_testA', 'refcoco+_bbox_testB', 'refcoco+_bbox_val', 'refcoco+_seg_testA',
                    'refcoco+_seg_testB', 'refcoco+_seg_val'],
                5: ['refcoco_bbox_test', 'refcoco_bbox_val', 'refcoco_bbox_testA', 'refcoco_bbox_testB',
                    'refcoco_seg_test', 'refcoco_seg_val'],
                6: ['refcoco_seg_testA', 'refcoco_seg_testB', 'refcocog_bbox_test', 'refcocog_bbox_val',
                    'refcocog_seg_test', 'refcocog_seg_val'],
                7: ['textcaps_val', 'textvqa_val', 'vizwiz_vqa_val', 'nocaps_val'],
                8: ['vqav2_val']
                }

datasets = id2benchmark[args.task]

for dataset in datasets:

    path_src = f'{args.source_dir}/{dataset}/'
    path_des = f'{args.des_dir}/{dataset}/'

    if not os.path.exists(path_des):
        os.makedirs(path_des)


    if 'llava' in dataset or dataset == 'mmvet':
        gpt_score = {}
    else:
        rouge_matrix = {}

    remove_ds_store(path_src)
    for model in os.listdir(path_src):
        print(model)

        with open(os.path.join(path_src, model)) as f:
            data = json.load(f)['logs']

        if 'llava' in dataset or dataset == 'mmvet':
            if 'model' not in gpt_score:
                gpt_score['model'] = []
        else:
            if 'model' not in rouge_matrix:
                rouge_matrix['model'] = []

        if 'llava' in dataset or dataset == 'mmvet':
            gpt_score['model'].append(model.replace('.json', ''))
        else:
            rouge_matrix['model'].append(model.replace('.json', ''))

        for index, entry in enumerate(data):
            id = entry['doc_id']
            key = f'{dataset}_{id}'
            print(key)

            if dataset == 'mmvet':
                if key not in gpt_score:
                    gpt_score[key] = []
                gpt_score[key].append(entry['gpt_eval_score']['score']) #already normalized
                continue

            elif 'llava' in dataset:
                if key not in gpt_score:
                    gpt_score[key] = []
                gpt_score[key].append(entry['gpt_eval_llava_all']['scores'][1] / 10.0)

            else:
                if key not in rouge_matrix:
                    rouge_matrix[key] = []

                if 'flickr' in dataset:
                    ground_truths = entry['doc']['caption']
                elif 'textcaps' in dataset:
                    ground_truths = entry['doc']["caption_str"]
                elif 'multidocvqa' in dataset:
                    ground_truths = entry['accuracy']['answer']
                else:
                    ground_truths = entry['target']
                if 'multidocvqa' in dataset:
                    prediction = entry['accuracy']['pred_answer']
                else:
                    prediction = entry['filtered_resps'][0]

                rouge = calculate_rouge(ground_truths, prediction)
                normalized_rouge = normalize_score(rouge)
                rouge_matrix[key].append(normalized_rouge)

    if dataset == 'mmvet' or 'llava' in dataset:
        save_score_to_parquet(gpt_score, 'gpt_score', path_des)
    else:
        save_score_to_parquet(rouge_matrix, 'rouge', path_des)

