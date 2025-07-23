from datasets import load_dataset
import torch
from transformers import DINOv2Model, DINOv2Processor
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


# Functions
def get_embeddings(image, text):
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.image_embeds, outputs.text_embeds


def normalize(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True)


def find_neighbors(embedding_idx, distances, cluster_assignments, difficulty="easy"):
    same_cluster = cluster_assignments == cluster_assignments[embedding_idx]
    different_clusters = ~same_cluster

    if difficulty == "easy":
        mask = same_cluster
    elif difficulty == "medium":
        neighboring_clusters = (
            cluster_assignments == (cluster_assignments[embedding_idx] - 1) % n_clusters
        ) | (
            cluster_assignments == (cluster_assignments[embedding_idx] + 1) % n_clusters
        )
        mask = neighboring_clusters
    elif difficulty == "hard":
        mask = different_clusters

    masked_distances = np.where(mask, distances[embedding_idx], np.inf)
    nearest_neighbor_idx = np.argmin(masked_distances)
    return nearest_neighbor_idx


dataset = load_dataset("lmms-lab/COCO-Caption2017")
val_data = dataset["val"]
test_data = dataset["test"]

print(test_data)

model = DINOv2Model.from_pretrained("facebook/dino-v2-base")
processor = DINOv2Processor.from_pretrained("facebook/dino-v2-base")

image_embeddings = []
text_embeddings = []

for item in val_data:
    image = item["image"]
    text = item["question"]

    image_emb, text_emb = get_embeddings(image, text)

    image_embeddings.append(image_emb)
    text_embeddings.append(text_emb)

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)

image_embeddings = normalize(image_embeddings)
text_embeddings = normalize(text_embeddings)

combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
combined_embeddings_np = combined_embeddings.cpu().numpy()
n_clusters = 500

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_embeddings_np)

cluster_assignments = kmeans.labels_

distances = cosine_distances(combined_embeddings_np)
associations = []

for i in range(len(combined_embeddings_np)):
    easy_neighbor = find_neighbors(i, distances, cluster_assignments, "easy")
    medium_neighbor = find_neighbors(i, distances, cluster_assignments, "medium")
    hard_neighbor = find_neighbors(i, distances, cluster_assignments, "hard")
    associations.append((easy_neighbor, medium_neighbor, hard_neighbor))
