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
                    3: ['mmbench_en_dev'],
                    4: ['refcoco+_bbox_testA', 'refcoco+_bbox_testB', 'refcoco+_bbox_val', 'refcoco+_seg_testA', 'refcoco+_seg_testB', 'refcoco+_seg_val'],
                    5: ['refcoco_bbox_test', 'refcoco_bbox_val', 'refcoco_bbox_testA', 'refcoco_bbox_testB', 'refcoco_seg_test', 'refcoco_seg_val'],
                    6: ['refcoco_seg_testA', 'refcoco_seg_testB', 'refcocog_bbox_test', 'refcocog_bbox_val', 'refcocog_seg_test', 'refcocog_seg_val'],
                    7: ['textcaps_val','nocaps_val'],
                    8: ['mmbench_en_test'],
                    }

    benchmarks = id2benchmark[args.task]

    for benchmark in benchmarks:
        print(f"Running benchmark: {benchmark}")
        benchmark_dir = f'{args.root_dir}/data/lmms_eval/{benchmark}'

        save_file = f'{args.root_dir}/results/pairwise_arena/{benchmark}.parquet'
        if os.path.exists(save_file):
            print("Results already exist, skipping")
            continue

        models = os.listdir(benchmark_dir)
        model_combinations = combinations(models, 2)

        print("Creating new pairwise ranking for ", benchmark)
        pairwise = {}
        pairwise['model_a'] = []
        pairwise['model_b'] = []
        pairwise['winner'] = []
        pairwise['sample_id'] = []

        for model1, model2 in model_combinations:
            mod1 = model1.replace('.json', '')
            mod2 = model2.replace('.json', '')

            print(f"Comparing {model1} and {model2}")
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


                for key_tie in keys_tie:
                    pairwise['model_a'].append(mod1)
                    pairwise['model_b'].append(mod2)
                    pairwise['winner'].append('tie')
                    pairwise['sample_id'].append(key_tie)

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
                    pairwise['model_a'].append(mod1)
                    pairwise['model_b'].append(mod2)
                    if score == 'A':
                        pairwise['winner'].append('model_a')
                    else:
                        pairwise['winner'].append('model_b')
                    pairwise['sample_id'].append(key)

        print("Saving results to ", save_file)
        df = pd.DataFrame(pairwise)
        df.to_parquet(save_file)

if __name__ == '__main__':
    main()





