import os
os.environ['HF_HOME'] = '/mnt/qb/work/bethge/bkr536/.cache/'
from itertools import combinations
import pandas as pd

from prometheus_eval import PrometheusEval
from prompt import PREFIX_PROMPT
from utils import *


root_dir = '/mnt/qb/work/bethge/bkr536/lifelong'
# root_dir = '/Users/heikekoenig/irp/lifelong_analysis/'

judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", relative_grade_template=PREFIX_PROMPT)

benchmark = 'coco2014_cap_val'
print(f"Running benchmark: {benchmark}")
benchmark_dir = f'{root_dir}/data/vlm_results/{benchmark}'

save_file = f'{root_dir}/results/pairwise/{benchmark}.parquet'

models = os.listdir(benchmark_dir)
model_combinations = combinations(models, 2)

pairwise = {}
for model1, model2 in model_combinations:
    if 'model' not in pairwise:
        pairwise['model'] = []
    models = model1.replace('.json','')+"_vs_"+model2.replace('.json','')
    pairwise['model'].append(models)


    print(f"The models being used are {model1} and {model2}")

    model1_path = os.path.join(benchmark_dir, model1)
    model2_path = os.path.join(benchmark_dir, model2)

    model1_results = load_results(model1_path)
    model2_results = load_results(model2_path)

    for logs1, logs2 in zip(model1_results, model2_results):
        id1 = logs1['doc_id']
        id2 = logs2['doc_id']
        assert id1 == id2
        key = f'{benchmark}_{id1}'
        if key not in pairwise:
            pairwise[key] = []

        answers, response1, response2 = compare_models(logs1, logs2)

        ground_truth = '; '.join(answers)


        data = {
            "instruction": "The ground truth captions of a provided image are listed here. They are separated by a semi-colon. Please carefully observe the image and come up with a caption for the image.",
            "response_A": response1,
            "response_B": response2,
            "reference_answer": f"Ground truth captions: {ground_truth}",
            "rubric": "Is the answer well correlated with the ground truth captions?"
        }

        feedback, score = judge.single_relative_grade(**data)

        print("Feedback:", feedback)
        print("Score:", score)

        pairwise[key].append(score)

df = pd.DataFrame(pairwise)
df.to_parquet(save_file)
