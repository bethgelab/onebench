import requests
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse

def check_website_existence(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False
def get_subdirectories(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            subdirectories = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('/') and not href.startswith('#'):
                    subdirectories.append(urljoin(url, href))
            return subdirectories
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def extract_dataset_and_model(text):
    prefix_match = re.search(r"^[^:]+", text)
    prefix = prefix_match.group(0) if prefix_match else None

    # Extract the model name
    model_match = re.search(r"model=([^,/]+)", text)
    model = model_match.group(1) if model_match else None

    # Attempt to extract the value after 'data_augmentation='
    data_augmentation_match = re.search(r"data_augmentation=([^,]+)", text)
    data_augmentation = data_augmentation_match.group(1) if data_augmentation_match else None

    # Extract the dataset
    dataset_match = re.search(r"(groups|subject|subset)=([^,/]+)", text)
    dataset = dataset_match.group(2) if dataset_match else None

    # Match LMMs-Eval
    if 'uw-madison_llava' in model:
        model = model.rstrip('-hf/').replace('uw-madison_', '').replace('-', '_').replace('v1.6', '1.6')

    # Combine prefix and data_augmentation if it exists, otherwise just use dataset
    if data_augmentation:
        result = f"{prefix}_{data_augmentation}", model
    elif "image2webpage" in text:
        result = f"{prefix}_css", model
    elif 'math_vista' in text:
        model = re.search(r"model=([^,]+)", text).group(1).rstrip('/') if re.search(r"model=([^,]+)", text) else None
        dataset = f"math_vista_{re.search(r'grade=([^,]+),question_type=([^,]+)', text).group(1)}_{re.search(r'grade=([^,]+),question_type=([^,]+)', text).group(2)}" if re.search(
            r"grade=([^,]+),question_type=([^,]+)", text) else None
        result = dataset, model
    elif 'pairs' in text:
        model = re.search(r"model=([^,]+)", text).group(1).rstrip('/') if re.search(r"model=([^,]+)", text) else None
        dataset = f"pairs_{match.group(1)}_{match.group(2)}" if (
            match := re.search(r"subset=(\w+),person=(\w+)", text)) else None
        result = dataset, model
    elif 'crossmodal_3600' in text:
        model = re.search(r"model=([^,/]+)", text).group(1).replace('-', '_')
        dataset = (
            f"{re.match(r'^[^:]+', text).group(0)}_{re.search(r'location=([^,]+)', text).group(1)}_{re.search(r'language=([^,]+)', text).group(1)}"
            if re.search(r'location=([^,]+)', text)
            else f"{re.match(r'^[^:]+', text).group(0)}_{re.search(r'language=([^,]+)', text).group(1)}")
        result = dataset, model
    else:
        result = f"{prefix}_{dataset}" if dataset else prefix, model

    return result

website_url = "https://nlp.stanford.edu/helm/vhelm/benchmark_output/runs/latest/"

save_root = 'data/vhelm'

id2benchmark = {0: ['a_okvqa'],
                1: ['bingo','gqa','flickr','pope'],
                2: ['crossmodal_3600'],
                3: ['hateful_memes','mementos','mme',],
                4: ['math_vista'],
                5: ['image2webpage','mscoco_captioning','mscoco_categorization','multipanel_vqa','originality_vlm'],
                6: ['pairs','seed_bench'],
                7: ['unicorn','viz_wiz','vqa']
                }

parser = argparse.ArgumentParser(description="Pairwise ranking using Prometheus2")
parser.add_argument(
        "--task", default=0,
        help="task number", type=int)
args = parser.parse_args()
data_run = id2benchmark[args.task]
mm_safety_bench_remove = ['hate_speech','illegal_activity','physical_harm','sex']
subdirectories = get_subdirectories(website_url)
for subdir in subdirectories:

    if not any(element in subdir for element in data_run):
        continue

    if "model=" not in subdir:
        continue
    print(subdir)

    # Remove T2I jsons
    if "max_eval_instances" in subdir:
        continue

    # Remove instagram augmentation
    if "mm_safety_bench" in subdir and any([x in subdir for x in mm_safety_bench_remove]):
        continue

    if "max_train_instances" in subdir:
        continue

    prompt_url = f"{subdir}/instances.json"
    score_url = f"{subdir}/per_instance_stats.json"

    # Check if the prompt and score files exist
    if not check_website_existence(prompt_url) or not check_website_existence(score_url):
        print(f"Prompt or score file not found for {subdir}")
        continue

    subdir_details = subdir.replace(website_url, '')
    dataset, model = extract_dataset_and_model(subdir_details)
    if dataset.endswith('/'):
        dataset = dataset[:-1]

    print(f"Saving prompts and score for {dataset} and {model}")

    prompt_dir = os.path.join(save_root, 'prompts')
    score_dir = os.path.join(save_root, 'score', dataset.lower())
    [os.makedirs(directory, exist_ok=True) for directory in [prompt_dir, score_dir]]

    prompt_file = os.path.join(prompt_dir, dataset.lower() + '.json')
    score_file = os.path.join(score_dir, model.lower() + '.json')
    print(score_file)

    if os.path.exists(score_file):
        print(f"Prompt and score files already exist for {model}")
        continue

    if not os.path.exists(prompt_file):
        # Send a GET request to the URL to download the JSON file
        response_prompt = requests.get(prompt_url)
        response_prompt.raise_for_status()  # Raise an exception for invalid HTTP responses

        with open(prompt_file, "wb") as f:
            f.write(response_prompt.content)

    # Send a GET request to the URL to download the JSON file
    response_score = requests.get(score_url)
    response_score.raise_for_status()  # Raise an exception for invalid HTTP responses


    with open(score_file, "wb") as f:
        f.write(response_score.content)
