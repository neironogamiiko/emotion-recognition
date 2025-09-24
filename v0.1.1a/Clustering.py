import cv2
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from hdbscan import HDBSCAN
from facenet_pytorch import InceptionResnetV1

from torch import nn

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")

import Utility

IMAGE_SIZE = (96,96)
BATCH_SIZE = 64

# data_path = Path("/home/frasero/PycharmProjects/Datasets/Emotions")
# output_path = Path("/home/frasero/PycharmProjects/Datasets/Emotions/Converted")

# Utility.convet2jpg(input_path=data_path,
#                    output_path=output_path)

dataset_path = Path("/home/frasero/PycharmProjects/Datasets/Emotions")

# <------ DATASET ------> #

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=str(dataset_path), transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# <------ FEATURES ------> #

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
with torch.inference_mode():
    for image_tensor, _ in data_loader:
        image_tensor = image_tensor.to(device)

        features = facenet(image_tensor)
        embeddings.append(features.cpu().numpy())
embeddings = numpy.concatenate(embeddings, axis=0)
print(f"Embedding's shape: {embeddings.shape}")

# <------ DIMENSION REDUCTION ------> #

# pca = PCA(n_components=50, random_state=42)
# embedding_pca = pca.fit_transform(embeddings)
# print(f"Shape after PCA: {embedding_pca.shape}")

# <------ K-MEANS CLUSTERING ------> #

kmeans = KMeans(n_clusters=5,
                random_state=42,
                n_init=10)

labels_kmeans = kmeans.fit_predict(embeddings)

Utility.clustering_metrics(embeddings=embeddings,
                           labels=labels_kmeans,
                           method_name="K-Means")

# <------ AGGLOMERATIVE CLUSTERING ------> #

agglomerative = AgglomerativeClustering(n_clusters=5,
                                        linkage='ward') # 'average', 'complete'

labels_agglomerative = agglomerative.fit_predict(embeddings)

Utility.clustering_metrics(embeddings=embeddings,
                           labels=labels_agglomerative,
                           method_name='Agglomerative')

# <------ HDBSCAN CLUSTERING ------> #

dbscan = HDBSCAN(min_cluster_size=5)
labels_dbscan = dbscan.fit_predict(embeddings)

Utility.clustering_metrics(embeddings=embeddings,
                           labels=labels_dbscan,
                           method_name='HDBSCAN')

# <------ ELBOW ------> #

# inertias = []
# K_range = range(2, 20)
# for k in K_range:
#     km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
#     inertias.append(km.inertia_)
#
# plt.plot(K_range, inertias, "o-")
# plt.xlabel("Number of clusters K")
# plt.ylabel("Inertia (Within-Cluster SSE)")
# plt.title("Elbow Method for Optimal K")
# plt.show()