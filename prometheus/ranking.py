import os
os.environ['HF_HOME'] = '/mnt/qb/work/bethge/bkr536/.cache/'
# task = int(os.environ['SLURM_PROCID'])

import argparse
from itertools import combinations, zip_longest
import pandas as pd

from prometheus_eval import PrometheusEval

from prompt import *
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Pairwise ranking using Prometheus2")

    parser.add_argument(
        "--root_dir", default="/mnt/qb/work/bethge/bkr536/lifelong",
        help="root directory", type=str)

    parser.add_argument(
        "--task", default=0,
        help="task number", type=int)

    parser.add_argument(
        "--batch_size", default=300,
        help="batch size", type=int)

    args = parser.parse_args()

    judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", relative_grade_template=PREFIX_PROMPT, dtype = 'bfloat16')

    id2benchmark = {0: ['coco2014_cap_val'],
                    1: ['coco2017_cap_val'],
                    2: ['flickr30k_test'],
                    3: ['mmbench_en_dev','mmbench_en_test'],
                    4: ['refcoco+_bbox_testA', 'refcoco+_bbox_testB', 'refcoco+_bbox_val', 'refcoco+_seg_testA', 'refcoco+_seg_testB', 'refcoco+_seg_val'],
                    5: ['refcoco_bbox_test', 'refcoco_bbox_val', 'refcoco_bbox_testA', 'refcoco_bbox_testB', 'refcoco_seg_test', 'refcoco_seg_val'],
                    6: ['refcoco_seg_testA', 'refcoco_seg_testB', 'refcocog_bbox_test', 'refcocog_bbox_val', 'refcocog_seg_test', 'refcocog_seg_val'],
                    7: ['textcaps_val','nocaps_val'],
                    }

    id2benchmark = {0: ['coco2014_cap_val', 'coco2017_cap_val','refcoco+_bbox_testA', 'refcoco+_bbox_testB', 'refcoco+_bbox_val', 'refcoco+_seg_testA', 'refcoco+_seg_testB', 'refcoco+_seg_val','refcoco_bbox_test', 'refcoco_bbox_testA', 'refcoco_bbox_testB', 'refcoco_seg_test', 'refcoco_seg_val','refcoco_seg_testA', 'refcoco_seg_testB', 'refcocog_bbox_test', 'refcocog_bbox_val', 'refcocog_seg_test', 'refcocog_seg_val'],
                    1: ['flickr30k_test'],
                    2: ['mmbench_en_dev','refcoco_bbox_val'],
                    3: ['textcaps_val','nocaps_val'],
                    }

    benchmarks = id2benchmark[args.task]

    for benchmark in benchmarks:
        print(f"Running benchmark: {benchmark}")
        benchmark_dir = f'{args.root_dir}/data/vlm_results/{benchmark}'

        save_file = f'{args.root_dir}/results/pairwise_num/{benchmark}_tie.parquet'

        models = os.listdir(benchmark_dir)
        model_combinations = combinations(models, 2)

        if os.path.exists(save_file):
            print("Loading existing parquet file")
            pairwise_df = pd.read_parquet(save_file)
            pairwise = pairwise_df.to_dict(orient='list')
            print("Existing model pairs")
            print(pairwise['model'])
        else:
            print("Creating new pairwise ranking for ", benchmark)
            pairwise = {}

        for model1, model2 in model_combinations:
            if 'model' not in pairwise:
                pairwise['model'] = []

            model_combo = model1.replace('.json', '') + "_vs_" + model2.replace('.json', '')
            model_combo_rev = model2.replace('.json', '') + "_vs_" + model1.replace('.json', '')

            if model_combo in pairwise['model'] or model_combo_rev in pairwise['model']:
                print(model_combo, "already exists")
                continue
            else:
                pairwise['model'].append(model_combo)
                print(f"The models being used are {model1} and {model2}")

            model1_path = os.path.join(benchmark_dir, model1)
            model2_path = os.path.join(benchmark_dir, model2)

            model1_results = load_results(model1_path)
            model2_results = load_results(model2_path)

            for batch1, batch2 in zip_longest(
                    (model1_results[i:i + args.batch_size] for i in range(0, len(model1_results), args.batch_size)),
                    (model2_results[i:i + args.batch_size] for i in range(0, len(model2_results), args.batch_size)),
                    fillvalue=[]):

                if benchmark == 'flickr30k_test':
                    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, _ = compare_models(
                        batch1, batch2, benchmark, metric='caption')
                elif benchmark == 'nocaps_val':
                    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, _ = compare_models(
                        batch1, batch2, benchmark, metric='annotations_captions')
                elif 'textcaps' in benchmark:
                    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, _ = compare_models(
                        batch1, batch2, benchmark, metric='reference_strs')
                elif 'mmbench' in benchmark:
                    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, arguments = compare_models(
                        batch1, batch2, benchmark, metric='answer')
                else:
                    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, _ = compare_models(
                        batch1, batch2, benchmark, metric='answer')

                for key in keys:
                    if key not in pairwise:
                        pairwise[key] = []

                for key_tie in keys_tie:
                    if key_tie not in pairwise:
                        pairwise[key_tie] = []
                    pairwise[key_tie].append(0.5)

                if 'mmbench' in benchmark:
                    data = {
                        "params": {},
                        "instructions": arguments,
                        "responses_A": model1_predictions,
                        "responses_B": model2_predictions,
                        "reference_answers": ground_truths,
                        "rubric": rubric_mmbench,
                    }

                else:

                    data = {
                        "params": {},
                        "instructions": [instruction_cap]*len(ground_truths),
                        "responses_A": model1_predictions,
                        "responses_B": model2_predictions,
                        "reference_answers": ground_truths,
                        "rubric": rubric_cap,
                    }

                feedbacks, scores = judge.relative_grade(**data)

                for i, key in enumerate(keys):
                    feedback = feedbacks[i]
                    score = scores[i]
                    print("Feedback:", feedback)
                    print("Score:", score)
                    if score == 'A':
                        pairwise[key].append(1)
                    else:
                        pairwise[key].append(0)

            print("Saving A matrix for ", model_combo)
            df = pd.DataFrame(pairwise)
            df.to_parquet(save_file)

if __name__ == '__main__':
    main()