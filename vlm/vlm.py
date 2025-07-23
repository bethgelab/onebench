import os
import json
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Saving JSON files for t2i from the HEIM website")

    parser.add_argument(
        "--path_src", default="/Users/heikekoenig/irp/lifelong_analysis/data/vlm/ai2d/",
        help="root directory of dataset results", type=str)

    parser.add_argument("--metric",
                        default='exact_match',
                        help="Which metric to use. Refer to the README for more", type=str)

    args = parser.parse_args()

    A_matrix = {}
    for model in os.listdir(args.path_src):

        with open(os.path.join(args.path_src, model)) as f:
            data = json.load(f)['logs']

        reference_dict = {}
        for index, entry in enumerate(data):
            doc_id = entry['doc_id']
            reference_dict[doc_id] = index

        model_results = populate_list_with_null(reference_dict)

        #Speed up the process for larger datasets
        doc_id_to_entry = {entry['doc_id']: entry for entry in data}

        for doc_id in reference_dict:
            index = reference_dict[doc_id]
            score = doc_id_to_entry.get(doc_id)[args.metric]
            model_results[index] = score

        A_matrix[model.replace('.json', '')] = model_results

    with open(os.path.join(args.path_src, os.path.basename(os.path.normpath(args.path_src)) + '_A_matrix.json'), 'w') as f:
        json.dump(A_matrix, f, indent=4)

if __name__ == '__main__':
    main()