import hashlib
import torch
import numpy as np
from PIL import Image
import re
import base64

def compute_base64(image):
    image_bytes = image.tobytes()
    return base64.b64encode(image_bytes)

def compute_image_hash(image):
    image_bytes = image.tobytes()
    return hashlib.md5(image_bytes).hexdigest()

def add_vector(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)

def embed_siglip(image, processor, model, device):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features

def compute_embedding(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy()

def extract_dataset_name(column_name):
    match = re.match(r"([a-zA-Z0-9_]+)_\d+", column_name)
    if match:
        return match.group(1)
    else:
        if "seedbench-2" in column_name:
            return "seedbench-2"
        else:
            return None

def get_text_embedding(text_prompt, processor, device):
    inputs = processor(text=[text_prompt], return_tensors="pt", padding=False, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    text_embedding = torch.nn.functional.normalize(outputs, dim=1) #NEW
    return text_embedding