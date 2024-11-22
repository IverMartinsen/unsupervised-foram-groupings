import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from utils import init_centroids_semi_supervised

df = pd.read_csv("Data/FEATURES_Gol-F-30-3, 19-20_zoom 35.csv")
path_to_crops = "Data/CROPS_Gol-F-30-3, 19-20_zoom 35/images"
destination = "Data/GROUPS_Gol-F-30-3, 19-20_zoom 35"

# sort df by filename
df = df.sort_values("filename").reset_index(drop=True)

try:
    X = df.drop(columns=["label", "filename"])
except KeyError:
    X = df.drop(columns=["filename"])

F = df["filename"]

os.makedirs(destination, exist_ok=True)

known_sediment =  [7, 50, 365, 426, 447]
known_benthic = [54, 106, 331]
known_planktic = [57, 439, 460]
known = np.array(known_sediment + known_benthic + known_planktic)

x_lab = np.array(X.loc[known])
y_lab = np.array([0] * len(known_sediment) + [1] * len(known_benthic) + [2] * len(known_planktic))
x_un = np.array(X.drop(known))

centroids, cluster_labs = init_centroids_semi_supervised(x_lab, y_lab, x_un, 10)

kmeans = KMeans(n_clusters=10, random_state=0, init=centroids)

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
    plt.savefig(f"{destination}/Group__with_knowns{i}.png")
    plt.close()
