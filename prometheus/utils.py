import json
import difflib

def load_results(file_path):
    with open(file_path) as f:
        data = json.load(f)['logs']
    return data


def compare_models(batch1, batch2, benchmark, ignore, metric = 'answer', threshold = 0.9):
    keys, keys_tie, ground_truths, model1_predictions, model2_predictions, arguments = [], [], [], [], [], []
    for logs1, logs2 in zip(batch1, batch2):
        id1 = logs1['doc_id']
        id2 = logs2['doc_id']
        assert id1 == id2

        key = f'{benchmark}_{id1}'
        if key in ignore:
            continue
        if 'mmbench' in benchmark:
            x = logs1['doc']['answer']
            ground_truth1 = logs1['doc'][x]
            y = logs2['doc']['answer']
            ground_truth2 = logs2['doc'][y]
        else:
            ground_truth1 = logs1['doc'][metric]
            ground_truth2 = logs2['doc'][metric]

        model1_prediction = logs1['filtered_resps'][0].strip()  # Assuming only one prediction
        model2_prediction = logs2['filtered_resps'][0].strip()  # Assuming only one prediction

        # assert ground_truth1 == ground_truth2

        argument = logs1['arguments'][0][0][3:]
        similarity_ratio = difflib.SequenceMatcher(None, model1_prediction, model2_prediction).ratio()

        if similarity_ratio >= threshold:
            keys_tie.append(key)
        else:
            keys.append(key)

            if 'mmbench' in benchmark:
                ground_truths.append(ground_truth1)
            else:
                ground_truths.append('; '.join(ground_truth1))
            model1_predictions.append(model1_prediction)
            model2_predictions.append(model2_prediction)
            arguments.append(argument)

    return keys, keys_tie, ground_truths, model1_predictions, model2_predictions, arguments
