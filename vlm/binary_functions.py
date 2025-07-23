import difflib
import string
def populate_list_with_null(index_dict):
    new_list = [None] * (len(index_dict.keys()) + 1)  # Initialize a list with None values

    for index in index_dict.values():
        if index <= len(index_dict.keys()) + 1:
            new_list[index] = None  # Populate the specified indexes with None

    return new_list


def variations(list_of_dicts):
    variations = set()
    for dictionary in list_of_dicts:
        if dictionary:  # Check if the dictionary is not empty
            last_key = list(dictionary.keys())[-1]
            variations.add(last_key)
    return list(variations)


def exact_match(dataset, entry):
    if dataset == 'textvqa_val' or dataset == 'vqav2_val':
        value = entry['exact_match']
    else:
        if len(entry['target'].lower()) == len(entry['filtered_resps'][0].lower().strip('.')):
            value = entry['exact_match']
        else:
            similarity_ratio = difflib_ratio(entry['target'].lower(),
                                             entry['filtered_resps'][0].lower().strip('answer: ').strip('.'))
            if similarity_ratio >= 0.5:
                value = 1.0
            else:
                value = 0.0

    return value


def difflib_ratio(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def iconqa(entry):
    choices = entry['doc']['choices']
    if choices == None:
        similarity_ratio = difflib_ratio(entry['doc']['answer'].lower(),
                                         entry['filtered_resps'][0].lower().strip('answer: ').strip('.'))
        if similarity_ratio >= 0.5:
            value = 1.0
        else:
            value = 0.0
    elif isinstance(choices, str):
        choices_list = choices.split(',')
        keys = [chr(ord('a') + i) for i in range(len(choices_list))]
        choices_dict = dict(zip(keys, choices_list))
        pred = entry['filtered_resps'][0]

        pred = pred.lower().replace(' ', '').replace('answer', '')
        translator = str.maketrans('', '', string.punctuation+string.digits)
        pred = pred.translate(translator)

        # Remove dangling spaces using strip()
        pred = pred.strip()
        try:
            if entry['doc']['answer'] == choices_dict[pred]:
                value = 1.0
            else:
                value = 0.0
        except KeyError:
            value = None
            print(entry['doc_id'])
    return value

def mmbench(entry):
    answer = entry['submission']['answer'].lower()
    prediction = entry['submission']['prediction']
    translator = str.maketrans('', '', string.punctuation + string.digits)
    prediction = prediction.translate(translator).replace('n','').replace(' ','').strip()
    if prediction == '':
        value = None
    else:

        prediction = prediction.lower().strip('.').replace(' ', '').replace('answer', '')

        prediction = prediction.strip()

        if len(answer) == len(prediction):
            value = 1 if answer == prediction else 0
        else:

            value = 1 if answer == prediction[0] else 0

    return value

