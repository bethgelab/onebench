import torch
from transformers import AutoProcessor, SiglipModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import argparse
import polars as pl
import re
import json
import base64
from io import BytesIO
from PIL import Image
from utils_embed import *

root_dir = '/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/'
cache_dir = "/p/scratch/ccstdl/ghosh4_juwelsbooster/.cache"
parser = argparse.ArgumentParser(description="Embedding generation with SigLip")

parser.add_argument(
    "--source_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/lmms_eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--parquet_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/vlm/lmms-eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--result_dir", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/results/siglip/lmms-eval/",
    help="root directory for saved results", type=str)

parser.add_argument(
    "--img2id_path", default="/p/scratch/ccstdl/ghosh4_juwelsbooster/lifelong/data/id2img_comb.json",
    help="root directory for saved results", type=str)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

vhelm = pl.read_parquet("data/vlm/vhelm/results.parquet")
embeddings = np.load('results/siglip/lmms-eval/image_embeddings.npy')

# Load model, processor, and tokenizer
model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir = cache_dir).to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-i18n",cache_dir=cache_dir)

image_embeddings = []
index = len(embeddings)

index2col_vhelm = {}

dataset_mapping = {
    "a_okvqa": ["a_okvqa_a_okvqa_base", "a_okvqa_chinese", "a_okvqa_dialect_prob=1.0_source=sae_target=aave",
                "a_okvqa_hindi","a_okvqa_instagram", "a_okvqa_robustness", "a_okvqa_spanish", "a_okvqa_swahili", "a_okvqa"],
    "bingo": ["bingo_factual", "bingo_i2i", "bingo_t2i", "bingo_ocr", "bingo_region"],
    "crossmodal_3600": ["crossmodal_3600_chinese_english", "crossmodal_3600_cusco_quechua_english","crossmodal_3600_english_chinese",
                        "crossmodal_3600_english_english","crossmodal_3600_english_hindi","crossmodal_3600_english_spanish",
                        "crossmodal_3600_hindi_english", "crossmodal_3600_maori_english","crossmodal_3600_spanish_english",
                        "crossmodal_3600_swahili_english", "crossmodal_3600_telugu_english"],
    "mathvista": ["math_vista_college_free_form", "math_vista_college_multi_choice", "math_vista_high_school_free_form",
                  "math_vista_high_school_multi_choice", "math_vista_daily_life_free_form", "math_vista_daily_life_multi_choice"],
    "mmmu": ['mmmu_psychology','mmmu_literature', 'mmmu_economics', 'mmmu_music', 'mmmu_energy_and_power','mmmu_physics','mmmu_biology',
             'mmmu_diagnostics_and_laboratory_medicine','mmmu_pharmacy','mmmu_architecture_and_engineering','mmmu_design','mmmu_chemistry',
             'mmmu_computer_science','mmmu_geography','mmmu_marketing','mmmu_sociology','mmmu_finance','mmmu_agriculture','mmmu_manage',
             'mmmu_art','mmmu_mechanical_engineering','mmmu_accounting','mmmu_art_theory','mmmu_public_health','mmmu_clinical_medicine','mmmu_materials',
             'mmmu_electronics','mmmu_history','mmmu_basic_medical_science'],
    "mscoco": ['mscoco_categorization', 'mscoco_captioning', 'mscoco_captioning_long'],
    "multipanelvqa":['multipanelvqa_real-world', 'multipanelvqa_synthetic'],
    "pairs": ['pairs_occupations_black_man','pairs_potential_crime_black_woman','pairs_potential_crime_white_woman','pairs_status_black_man',
              'pairs_occupations_black_woman','pairs_occupations_white_woman','pairs_potential_crime_white_man','pairs_status_black_woman',
              'pairs_status_white_woman','pairs_status_white_man','pairs_occupations_white_man','pairs_potential_crime_black_man'],
    "seed_bench":['seed_bench_visual-reasoning', 'seed_bench_instance-interaction', 'seed_bench'],
    "unicorn": ['unicorn_oodcv-vqa', 'unicorn_sketchy-vqa', 'unicorn'],
    "vqa": ['vqa_robustness','vqa_spanish','vqa_dialect_prob=1.0_source=sae_target=aave','vqa_vqa_base','vqa_hindi','vqa_chinese']
}
lmms_mapping = {
    "a_okvqa": "coco2017_cap_val",
    "flickr30k": "flickr30k_test",
    "mathvista": "mathvista_testmini",
    "mmmu": "mmmu_val",
    "pope": "coco2014_cap_val",
    "seed_bench":"seedbench",
    "viz_wiz": "vizwiz_vqa_val",
    "vqa": "coco2014_cap_val"
}

with open(f'results/siglip/lmms-eval/index2col.json') as f:
    index2col = json.load(f)

#A-OKVQA
dataset = "a_okvqa"
with open(f'data/vhelm/prompts/{dataset}.json') as f:
    ds_vhelm = json.load(f)
with open(f'data/aokvqa/aokvqa_v1p0_val.json') as f:
    ds_info = json.load(f)
with open(f'data/lmms_eval/coco2017_cap_val/llava_1.6_13b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {int(entry["doc"]["file_name"].replace(".jpg", "").lstrip('0')): entry['doc_id'] for entry in ds_lmms["logs"]}
image_ids = {entry['question_id']: entry['image_id'] for entry in ds_info}

for i in range(len(ds_vhelm)):
    matrix_id = ds_vhelm[i]["id"].replace("id", "")
    if matrix_id not in vhelm.columns:
        continue
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1].replace(
        ".jpg", "")
    col_ids = [subds + f"_{matrix_id}" for subds in dataset_mapping[dataset]]
    col_ids = [element for element in col_ids if element in vhelm.columns]

    img_id = image_ids[filename]
    lmms_id = lmms_ids[img_id]
    lmms_col = lmms_mapping[dataset] + f"_{lmms_id}"
    idx = [key for key, val in index2col.items() if val == lmms_col]
    if idx == []:
        continue
    index = idx[0]
    if index not in index2col_vhelm.keys():
        index2col_vhelm[index] = col_ids
    else:
        index2col_vhelm[index].extend(col_ids)


# BINGO
dataset = "bingo"
dataset_images = f"data/{dataset}/"
for bingo_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{bingo_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = bingo_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
        print(filename)
        print(matrix_id)
        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")

#CROSSMODAL
dataset = "crossmodal_3600"
dataset_images = f"data/{dataset}/"
for crossmodal_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{crossmodal_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = crossmodal_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][1]["location"]
        filename = filename.replace("benchmark_output/scenarios/crossmodal_3600/images/","data/crossmodal_3600/")
        print(filename)
        print(matrix_id)

        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1
print(f"Total embeddings: {len(image_embeddings)}")

#FLICKR

dataset = "flickr30k"
with open(f'data/vhelm/prompts/{dataset}.json') as f:
    ds_vhelm = json.load(f)
with open(f'data/lmms_eval/flickr30k_test/llava_7b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["doc"]["filename"]: entry['doc_id'] for entry in ds_lmms["logs"]}
# image_ids = {entry['question_id']: entry['image_id'] for entry in ds_info}

for i in range(len(ds_vhelm)):
    matrix_id = dataset+"_"+ds_vhelm[i]["id"].replace("id", "")
    if matrix_id not in vhelm.columns:
        continue
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
    lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
    idx = [key for key, val in index2col.items() if val == lmms_col]
    if idx == []:
        continue
    index = idx[0]
    print(index)
    if index not in index2col_vhelm.keys():
        index2col_vhelm[index] = matrix_id
    else:
        index2col_vhelm[index] = [index2col_vhelm[index]]
        index2col_vhelm[index].extend(matrix_id)

print(f"Total embeddings: {len(image_embeddings)}")

#HATEFUL MEMES
dataset = "hateful_memes"
dataset_images = f"data/{dataset}/"
with open(f'data/vhelm/prompts/{dataset}.json') as f:
    ds_vhelm = json.load(f)
for i in range(len(ds_vhelm)):
    matrix_id = dataset+"_"+ds_vhelm[i]["id"].replace("id","")
    if matrix_id not in vhelm.columns:
        continue
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
    filename = filename.replace("benchmark_output/scenarios/","data/")
    print(filename)
    print(matrix_id)

    image = Image.open(filename).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
    image_embeddings.append(siglip_features)
    index2col_vhelm[index] = matrix_id
    print(f"{matrix_id} - {index}")
    index += 1
print(f"Total embeddings: {len(image_embeddings)}")

#MATHVISTA
dataset = "mathvista"
with open(f'data/lmms_eval/mathvista_testmini/idefics2_8b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["doc"]["image"].replace("images/",""): entry['doc_id'] for entry in ds_lmms["logs"]}
# image_ids = {entry['question_id']: entry['image_id'] for entry in ds_info}

for mathvista_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{mathvista_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = mathvista_ds+"_"+ds_vhelm[i]["id"].replace("id", "")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][1]["location"].split("/")[-1]
        lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
        idx = [key for key, val in index2col.items() if val == lmms_col]
        if idx == []:
            continue
        index = idx[0]
        print(index)
        if index not in index2col_vhelm.keys():
            index2col_vhelm[index] = matrix_id
        else:
            index2col_vhelm[index] = [index2col_vhelm[index]]
            index2col_vhelm[index].extend(matrix_id)


# Mementos
dataset = "mementos"
dataset_images = f"data/{dataset}/"

for mementos_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{mementos_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = mementos_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
        filename = filename.replace("benchmark_output/scenarios/","data/")
        print(filename)
        print(matrix_id)
        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")


#MME
dataset = "mme"
with open(f'data/lmms_eval/{dataset}/idefics2_8b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {}
for entry in ds_lmms["logs"]:
    question_id = entry["doc"]["question_id"].replace("images/", "")
    if question_id not in lmms_ids:
        lmms_ids[question_id] = [entry['doc_id']]
    else:
        lmms_ids[question_id].append(entry['doc_id'])

for mme_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{mme_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = mme_ds+"_"+ds_vhelm[i]["id"].replace("id", "")
        if matrix_id not in vhelm.columns:
            print(matrix_id)
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
        filename = filename.replace("-","/").replace(".png",".jpg")
        lmms_col = [lmms_mapping[dataset] + f"_{name}" for name in lmms_ids[filename]]
        print(lmms_col)
        for col in lmms_col:
            idx = [key for key, val in index2col.items() if val == col]

            if idx == []:
                continue
            index = idx[0]
            print(index)
            if index not in index2col_vhelm:
                index2col_vhelm[index] = [matrix_id]
            else:
                if isinstance(index2col_vhelm[index], list):
                    index2col_vhelm[index].append(matrix_id)
                else:
                    index2col_vhelm[index] = [index2col_vhelm[index], matrix_id]

#MMMU
dataset = "mmmu"
with open(f'data/lmms_eval/mmmu_val/idefics2_8b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["doc"]["id"]: entry['doc_id'] for entry in ds_lmms["logs"]}

for mmmu_ds in dataset_mapping[dataset]:
    print(mmmu_ds)
    with open(f'data/vhelm/prompts/{mmmu_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = mmmu_ds+"_"+ds_vhelm[i]["id"].replace("id", "")
        if matrix_id not in vhelm.columns:
            print(matrix_id)
            continue
        # filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][1]["location"].split("/")[-1]
        if len(ds_vhelm[i]["input"]["multimedia_content"]['media_objects']) == 1:
            filename = ds_vhelm[i]["references"][0]["output"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
        else:
            if "location" in ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][1].keys():
                filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][1]["location"].split("/")[-1]
            else:
                filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
        filename = filename.replace("-","/").replace("_image_1.png","").replace("_image_2.png","")
        lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
        print(lmms_col)
        idx = [key for key, val in index2col.items() if val == lmms_col]
        if idx == []:
            continue
        index = idx[0]
        print(index)
        if index not in index2col_vhelm.keys():
            index2col_vhelm[index] = matrix_id
        else:
            index2col_vhelm[index] = [index2col_vhelm[index]]
            index2col_vhelm[index].extend(matrix_id)

#COCO
dataset = "mscoco"

for coco_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{coco_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = mementos_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        base64_str = ds_vhelm[i]["input"]["text"].split("base64,")[1].split('"')[0]
        image_data = base64.b64decode(base64_str)  # im_bytes is a binary image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")

#multipanelvqa
dataset = "multipanelvqa"
dataset_images = f"data/{dataset}/"

for multi_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{multi_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = multi_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
        filename = filename.replace("benchmark_output/scenarios/","data/")
        print(filename)
        print(matrix_id)
        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")

#pairs
dataset = "pairs"
dataset_images = f"data/{dataset}/"

for pairs_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{pairs_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = pairs_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
        filename = filename.replace("benchmark_output/scenarios/","data/")
        filename = filename.replace("_black","/black")
        filename = filename.replace("_white","/white")

        print(filename)
        print(matrix_id)
        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")

#POPE
index2col_vhelm = {}
dataset = "pope"
with open(f'data/lmms_eval/{lmms_mapping[dataset]}/llava_7b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["doc"]["question_id"]: entry['doc_id'] for entry in ds_lmms["logs"]}

with open(f'data/vhelm/prompts/{dataset}.json') as f:
    ds_vhelm = json.load(f)
for i in range(len(ds_vhelm)):
    matrix_id = dataset+"_"+ds_vhelm[i]["id"].replace("id", "")
    if matrix_id not in vhelm.columns:
        # print(matrix_id)
        continue
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
    print(filename)
    lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
    print(lmms_col)
    idx = [key for key, val in index2col.items() if val == lmms_col]
    if idx == []:
        continue
    index = idx[0]
    if index not in index2col_vhelm:
        index2col_vhelm[index] = [matrix_id]
    else:
        if isinstance(index2col_vhelm[index], list):
            index2col_vhelm[index].append(matrix_id)
        else:
            index2col_vhelm[index] = [index2col_vhelm[index], matrix_id]

#SEEDBENCH
dataset = "seed_bench"
with open(f'data/lmms_eval/{lmms_mapping[dataset]}/idefics2_8b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["doc"]["question_id"]: entry['doc_id'] for entry in ds_lmms["logs"]}

for seed_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{seed_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = seed_ds+"_"+ds_vhelm[i]["id"].replace("id", "")
        if matrix_id not in vhelm.columns:
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1].replace(".png","")
        lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
        idx = [key for key, val in index2col.items() if val == lmms_col]
        if idx == []:
            continue
        index = idx[0]
        print(index)
        if index not in index2col_vhelm.keys():
            index2col_vhelm[index] = matrix_id
        else:
            index2col_vhelm[index] = [index2col_vhelm[index]]
            index2col_vhelm[index].extend(matrix_id)

#UNICORN
dataset = "unicorn"
dataset_images = f"data/{dataset}/"
for uni_ds in dataset_mapping[dataset]:
    with open(f'data/vhelm/prompts/{uni_ds}.json') as f:
        ds_vhelm = json.load(f)
    for i in range(len(ds_vhelm)):
        matrix_id = uni_ds+"_"+ds_vhelm[i]["id"].replace("id","")
        if matrix_id not in vhelm.columns:
            print(matrix_id)
            continue
        filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"]
        filename = filename.replace("benchmark_output/scenarios/","data/")
        if "ood" not in uni_ds:
            filename = filename.replace("_", "/", 1)
        print(filename)
        print(matrix_id)
        image = Image.open(filename).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        siglip_features = embed_siglip(image, processor, model, device).cpu().numpy()
        image_embeddings.append(siglip_features)
        index2col_vhelm[index] = matrix_id
        print(f"{matrix_id} - {index}")
        index += 1

print(f"Total embeddings: {len(image_embeddings)}")

#viz-wiz
dataset = "viz_wiz"
index2col_vhelm = {}
with open(f'data/lmms_eval/{lmms_mapping[dataset]}/llava_7b.json') as f:
    ds_lmms = json.load(f)
lmms_ids = {entry["submission"]["image"]: entry['doc_id'] for entry in ds_lmms["logs"]}
with open(f'data/vhelm/prompts/{dataset}.json') as f:
    ds_vhelm = json.load(f)
for i in range(len(ds_vhelm)):
    matrix_id = dataset+"_"+ds_vhelm[i]["id"].replace("id","")
    if matrix_id not in vhelm.columns:
        print(matrix_id)
        continue
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
    lmms_col = lmms_mapping[dataset] + f"_{lmms_ids[filename]}"
    idx = [key for key, val in index2col.items() if val == lmms_col]
    if idx == []:
        print(matrix_id)
        continue
    index = idx[0]
    print(index)
    if index not in index2col_vhelm.keys():
        index2col_vhelm[index] = matrix_id
    else:
        index2col_vhelm[index] = [index2col_vhelm[index]]
        index2col_vhelm[index].extend(matrix_id)

#VQA
dataset = "vqa"
with open(f'data/lmms_eval/coco2014_cap_val/llava_1.6_13b.json') as f:
    ds_lmms = json.load(f)
with open(f'data/vhelm/prompts/vqa_vqa_base.json') as f:
    ds_vhelm = json.load(f)
lmms_ids = {entry["doc"]["question_id"]: entry['doc_id'] for entry in ds_lmms["logs"]}

for i in range(len(ds_vhelm)):
    matrix_id = ds_vhelm[i]["id"].replace("id", "")
    filename = ds_vhelm[i]["input"]["multimedia_content"]['media_objects'][0]["location"].split("/")[-1]
    print(filename)
    col_ids = [subds + f"_{matrix_id}" for subds in dataset_mapping[dataset]]
    col_ids = [element for element in col_ids if element in vhelm.columns]

    lmms_id = lmms_ids[filename]
    lmms_col = lmms_mapping[dataset] + f"_{lmms_id}"
    print(lmms_col)
    idx = [key for key, val in index2col.items() if val == lmms_col]
    print(idx)
    if idx == []:
        continue
    index = idx[0]
    if index not in index2col_vhelm.keys():
        index2col_vhelm[index] = col_ids
    else:
        index2col_vhelm[index].extend(col_ids)

image_embeddings = np.concatenate(image_embeddings, axis=0)

combined_embeddings = np.concatenate((embeddings, image_embeddings), axis=0)
with open(f"{args.result_dir}index2col_vhelm.json", 'w') as f:
    json.dump(index2col_vhelm, f)

np.save(f"{args.result_dir}image_embeddings_combined.npy", combined_embeddings)