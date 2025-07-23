import os
from itertools import combinations
import pandas as pd

import difflib
from utils import load_results
tasks = range(8)
threshold = 0.95
root_dir = '/Users/heikekoenig/irp/lifelong_analysis/'


id2benchmark = {0: ['coco2014_cap_val'],
                1: ['coco2017_cap_val'],
                2: ['flickr30k_test'],
                3: ['mmbench_en_dev', 'mmbench_en_test'],
                4: ['refcoco+_bbox_testA', 'refcoco+_bbox_testB', 'refcoco+_bbox_val', 'refcoco+_seg_testA',
                    'refcoco+_seg_testB', 'refcoco+_seg_val'],
                5: ['refcoco_bbox_test', 'refcoco_bbox_val', 'refcoco_bbox_testA', 'refcoco_bbox_testB',
                    'refcoco_seg_test', 'refcoco_seg_val'],
                6: ['refcoco_seg_testA', 'refcoco_seg_testB', 'refcocog_bbox_test', 'refcocog_bbox_val',
                    'refcocog_seg_test','refcocog_seg_val'],
                7: ['textcaps_val', 'textcaps_test', 'nocaps_val'],
                }

for task in tasks:
    benchmarks = id2benchmark[task]

    for benchmark in benchmarks:
        print(f"Running benchmark: {benchmark}")
        benchmark_dir = f'{root_dir}/data/vlm_results/{benchmark}'

        load_file = f'{root_dir}/results/pairwise/{benchmark}.parquet'
        save_file = f'{root_dir}/results/pairwise_num/{benchmark}_tie.parquet'

        df = pd.read_parquet(save_file)
        df.set_index(df.columns[0], inplace=True)
        df.to_parquet(save_file)

        if os.path.exists(save_file):
            print("File exists")
            continue

        models = os.listdir(benchmark_dir)
        model_combinations = combinations(models, 2)

        df = pd.read_parquet(load_file)

        df.set_index(df.columns[0], inplace=True)

        # Iterate over the rows and columns to count wins and losses
        for index, row in df.iterrows():
            model_pair = index.split('_vs_')
            model_a = model_pair[0]
            model_b = model_pair[1]

            a_logs = load_results(f"data/vlm_results/{benchmark}/{model_a}.json")
            a_index = {document['doc_id']: document for document in a_logs}
            b_logs = load_results(f"data/vlm_results/{benchmark}/{model_b}.json")
            b_index = {document['doc_id']: document for document in b_logs}

            for col in df.columns:
                doc_id = int(col.split("_")[-1])
                model1_prediction = a_index[int(doc_id)]['filtered_resps'][0].lower().strip()
                model2_prediction = b_index[int(doc_id)]['filtered_resps'][0].lower().strip()
                similarity_ratio = difflib.SequenceMatcher(None, model1_prediction, model2_prediction).ratio()

                if similarity_ratio >= threshold:
                    df.at[index, col] = 0.5
                    print(doc_id)
                    print(f"Model 1 prediction: {model1_prediction}")
                    print(f"Model 2 prediction: {model2_prediction}")
                else:
                    if row[col] == 'A':
                        df.at[index, col] = 1
                    else:
                        df.at[index, col] = 0

        print("Saving A matrix for ", benchmark)
        df.to_parquet(save_file)