import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

df = pd.read_csv("/Users/ima029/Desktop/Unsupervised foraminifera groupings/features.csv")
path_to_crops = "/Users/ima029/Desktop/Unsupervised foraminifera groupings/imagefolder/images"

kmeans = KMeans(n_clusters=10, random_state=0)

group_labels = kmeans.fit_predict(df.drop(columns=["label", "filename"]))

for i in range(10):
    group = df[group_labels == i]
    filenames = group["filename"]
    n = np.sqrt(len(filenames))
    n = int(n) + 1 if n % 1 != 0 else int(n)
    fig, ax = plt.subplots(n, n, figsize=(20, 20))
    for j, filename in enumerate(filenames):
        img = Image.open(f"{path_to_crops}/{filename}").resize((224, 224))
        ax.flatten()[j].imshow(img)
    for ax in ax.flatten():
        ax.axis("off")
    plt.savefig(f"Group_{i}.png")
