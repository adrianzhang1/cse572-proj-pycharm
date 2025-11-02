from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TRIAL = "dim8-k300"
# OUTPUT_NAME = "k-10-hypercluster-k300"
OUTPUT_NAME = TRIAL

feature_vectors = pd.read_csv(f"clustering_{TRIAL}_features.csv").iloc[:, 1:].to_numpy()
#
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)
print(feature_vectors)
from sklearn.decomposition import PCA

# Reduce dimensionality from 512 to 50 components
pca = PCA(n_components=50)
feature_vectors = pca.fit_transform(feature_vectors)


# K_MAX = 2000
# K_STEP = 1
# K_MIN = 2
# from sklearn.metrics import silhouette_score
# sil = []
# max_score = 0
#
# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# for k in range(K_MIN, K_MAX+1,K_STEP):
#   kmeans = KMeans(n_clusters = k,init='k-means++').fit(feature_vectors)
#   labels = kmeans.labels_
#   current_score = silhouette_score(feature_vectors, labels, metric = 'euclidean')
#   sil.append(current_score)
#   print(f"k={k} score: {current_score}")
#   if current_score > max_score:
#       max_k = k
#       max_score = current_score
# print(f"max k:{max_k}")

km = KMeans(n_clusters=300, init='k-means++')
km.fit(feature_vectors)
#



clusters = pca.inverse_transform(km.cluster_centers_)
clusters = scaler.inverse_transform(clusters)
# # clusters = km.cluster_centers_
#
np.set_printoptions(suppress=True, precision=2)
#
# # print(clusters[0].reshape(EMBED_DIM, EMBED_DIM, EMBED_DIM))
# print(km.labels_)
#
#
# # clusters = clusters*100
# # clusters = clusters.astype(int)
#
print("Saving data")
data = pd.DataFrame(clusters)
print(data.head())
data.to_csv(f"clusters_{OUTPUT_NAME}.csv")

cluster_map = defaultdict(list)
for idx, label in enumerate(km.labels_):
    cluster_map[label].append(idx)

# Save to a .txt file
with open(f"clusters_{OUTPUT_NAME}_indices.txt", "w") as f:
    for cluster_id in sorted(cluster_map.keys()):
        indices = ", ".join(map(str, cluster_map[cluster_id]))
        f.write(f"{cluster_id}:\t{indices}\n")
