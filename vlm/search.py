import torch
import numpy as np
import json
from transformers import SiglipProcessor, SiglipModel, AutoTokenizer
import argparse
import os
import sys
from logger import Logger
from utils_embed import *
import polars as pl
from PIL import Image
from rank_imgs import return_retrievals2

parser = argparse.ArgumentParser(description="Embedding generation with SigLip")

parser.add_argument(
    "--source_dir", default="data/lmms_eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--parquet_dir", default="data/vlm/lmms-eval/",
    help="root directory to load results", type=str)

parser.add_argument(
    "--result_dir", default="results/siglip/lmms-eval/",
    help="root directory for saved results", type=str)

parser.add_argument(
    "--cache_dir", default="/mnt/qb/work/bethge/bkr536/.cache/huggingface/",
    help="root directory for saved results", type=str)

parser.add_argument(
    "--query_images", default="query_images_20",
    help="text prompt to query", type=str)

args = parser.parse_args()
root_dir = ''
folder_path = f'{args.result_dir}/query_images/'
sys.stdout = Logger(
        os.path.join(
            f"results/siglip/{args.query_images}.log"
        )
    )

with open(f'{root_dir}/results/siglip/lmms-eval/index2col.json') as f:
    index_to_col = json.load(f)
print("Loaded index2col")

with open(f'{root_dir}/data/id2img_comb.json') as f:
    col_to_ds = json.load(f)

# binary = os.listdir(f'{args.parquet_dir}/binary/')
# binary = [file_name for file_name in binary if file_name != '.DS_Store']
# binary = [re.sub(r'\.parquet$', '', file_name) for file_name in binary]

# numeric = os.listdir(f'{args.parquet_dir}/numeric/')
# numeric = [file_name for file_name in numeric if file_name != '.DS_Store']
# numeric = [re.sub(r'\.parquet$', '', file_name) for file_name in numeric]

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# Load the model, processor, and tokenizer
model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir=args.cache_dir).to(device)
processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-i18n", cache_dir=args.cache_dir)
print("Loaded model, processor and tokenizer")

embeddings = np.load(f'{root_dir}results/siglip/lmms-eval/image_embeddings.npy')
embeddings = torch.tensor(embeddings).to(device)
embeddings = torch.nn.functional.normalize(embeddings, dim=1)  # NEW
print("Loaded embeddings")

# df1 = pl.read_parquet(f'{args.parquet_dir}/binary.parquet')
# print("Loaded binary")
# df2 = pl.read_parquet(f'{args.parquet_dir}/numeric.parquet')
# print("Loaded numeric")
# a_matrix = df1.join(df2, on="model", how="inner")
# print("Loaded a-matrix")
for filename in os.listdir(folder_path):
    if filename.endswith(('png', 'jpg', 'jpeg')):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        processed_image = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_output = model.get_image_features(**processed_image)
        image_embedding = torch.nn.functional.normalize(image_output, dim=1)

        caption = os.path.splitext(filename)[0]
        text_prompt = caption.replace('_', ' ')
        text_prompt = f'This is a photo of {text_prompt}.'
        print(text_prompt)
        text_embedding = get_text_embedding(text_prompt, model, processor, device)

        img_dot_products = torch.matmul(embeddings, image_embedding.T).squeeze()  # Compute dot products
        img_top10_indices = torch.argsort(img_dot_products, descending=True)[:20]  # Get top 10 indices
        img_indices = img_top10_indices.cpu().numpy().tolist()  # Convert indices to a list
        print("Top 10 image indices:", img_indices)

        txt_dot_products = torch.matmul(embeddings, text_embedding.T).squeeze()
        txt_top10_indices = torch.argsort(txt_dot_products, descending=True)[:20]
        txt_indices = txt_top10_indices.cpu().numpy().tolist()

        print("Top 10 text indices:", txt_indices)

        # return_retrievals(img_indices, args, index_to_col, a_matrix, text_prompt, col_to_ds, emb_type='img')
        # return_retrievals(txt_indices, args, index_to_col, a_matrix, text_prompt, col_to_ds, emb_type='txt')


        return_retrievals2(img_indices, args, index_to_col, caption, col_to_ds, emb_type='img')
        return_retrievals2(txt_indices, args, index_to_col, caption, col_to_ds, emb_type='txt')
