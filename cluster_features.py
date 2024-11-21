import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

df = pd.read_csv("Data/FEATURES_Gol-F-30-3, 19-20_zoom 35.csv")
path_to_crops = "Data/CROPS_Gol-F-30-3, 19-20_zoom 35/images"
destination = "Data/GROUPS_Gol-F-30-3, 19-20_zoom 35"

os.makedirs(destination, exist_ok=True)

kmeans = KMeans(n_clusters=10, random_state=0)

try:
    X = df.drop(columns=["label", "filename"])
except KeyError:
    X = df.drop(columns=["filename"])

F = df["filename"]

group_labels = kmeans.fit_predict(X)

for i in range(10):
    filenames = F[group_labels == i]
    features = X[group_labels == i]
    centroid = kmeans.cluster_centers_[i]
    mean_distance = np.mean(np.linalg.norm(features - centroid, axis=1))
    n = np.sqrt(len(filenames))
    n = int(n) + 1 if n % 1 != 0 else int(n)
    fig, ax = plt.subplots(n, n, figsize=(20, 20))
    for j, filename in enumerate(filenames):
        img = Image.open(f"{path_to_crops}/{filename}").resize((224, 224))
        ax.flatten()[j].imshow(img)
    for ax in ax.flatten():
        ax.axis("off")
    fig.suptitle(f"Group {i} - Mean Distance: {mean_distance:.2f}", fontsize=20)
    plt.savefig(f"{destination}/Group_{i}.png")
    plt.show()
