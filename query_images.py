import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

df = pd.read_csv("Data/FEATURES_Gol-F-30-3, 19-20_zoom 35.csv")
path_to_crops = "Data/CROPS_Gol-F-30-3, 19-20_zoom 35/images"
destination = "Data/GROUPS_Gol-F-30-3, 19-20_zoom 35"

# sort df by filename
df = df.sort_values("filename").reset_index(drop=True)

os.makedirs(destination, exist_ok=True)

try:
    X = df.drop(columns=["label", "filename"])
except KeyError:
    X = df.drop(columns=["filename"])

F = df["filename"]

# pick 10 random images
np.random.seed(0)
random_indices = np.random.choice(len(X), 10, replace=False)
Q = X.iloc[random_indices]
D = pairwise_distances(Q, X)
NN = np.argsort(D, axis=1)[:, 1:11]

# display the images
fig, ax = plt.subplots(10, 11, figsize=(20, 20))
for i, (query, neighbors) in enumerate(zip(Q.iterrows(), NN)):
    query_img = Image.open(f"{path_to_crops}/{F.iloc[query[0]]}").resize((224, 224))
    ax[i, 0].imshow(query_img)
    ax[i, 0].set_title(f"Query {i}")
    ax[i, 0].axis("off")
    for j, neighbor in enumerate(neighbors):
        img = Image.open(f"{path_to_crops}/{F.iloc[neighbor]}").resize((224, 224))
        ax[i, j + 1].imshow(img)
        ax[i, j + 1].set_title(f"D = {D[i, neighbor]:.2f}")
        ax[i, j + 1].axis("off")
plt.savefig(f"{destination}/NN.png")

NN = np.argsort(D, axis=1)[1, 1:101]

fig, ax = plt.subplots(10, 10, figsize=(20, 20))
for i, neighbor in enumerate(NN):
    img = Image.open(f"{path_to_crops}/{F.iloc[neighbor]}").resize((224, 224))
    ax[i // 10, i % 10].imshow(img)
    ax[i // 10, i % 10].axis("off")
plt.savefig(f"{destination}/NN_1.png")

D = pairwise_distances(X, X)

# 7, 50, 365, 426, 447
query = 7
indices = np.array(query).flatten()
n = 4
num_steps = 7
distance_threshold = 75

for _ in range(num_steps):

    D_ = D[indices]
    NN = np.argsort(D_, axis=1)[:, :n]

    new_indices = list(set(list(indices) + list(NN.flatten())))

    if len(new_indices) == len(indices):
        print("No new images to add")
        break
    indices = new_indices

# also add 10 nearest neighbors
NN = np.argsort(D, axis=1)[query, 1:11]
indices = list(set(list(indices) + list(NN.flatten())))

NN = np.where(D[query, :] < distance_threshold)[-1]
indices = list(set(list(indices) + list(NN)))

# sort by distance to query
indices = np.array(indices)[D[indices, query].argsort()]

fig, ax = plt.subplots(10, 10, figsize=(20, 20))
for i, neighbor in enumerate(indices):
    img = Image.open(f"{path_to_crops}/{F.iloc[neighbor]}").resize((224, 224))
    ax[i // 10, i % 10].imshow(img)
    #ax[i // 10, i % 10].axis("off")
    #if neighbor == query:
    #    ax[i // 10, i % 10].set_title("Query")
    #else:
    #ax[i // 10, i % 10].set_title(f"D = {D[query, neighbor]:.2f}")
for ax in ax.flatten():
    ax.axis("off")

plt.savefig(f"{destination}/NN_{query}_{num_steps}_steps.png")
plt.close()