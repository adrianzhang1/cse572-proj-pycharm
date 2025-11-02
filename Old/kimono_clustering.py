from collections import defaultdict
from math import floor
# from oklch_conv import rgb_direct_oklch
from oklab_conv import srgb_to_oklab,oklab_to_srgb

# Load image and display
# img_original = cv2.imread(image_path)
# print('Dimensions : ',img_original.shape)
# print(img)
import math

import pandas
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import random

from sklearn.preprocessing import StandardScaler

TRIAL = "dim8-k300"
N_CLUSTERS = 300
GENERATE_VISUALS = True
EMBED_DIM = 8

# km = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++')

df_cols_color = [f'color_{i + 1}_{ch}' for i in range(N_CLUSTERS) for ch in ['L', 'a', 'b']]
df_cols_prop = [f'p_{i + 1}' for i in range(N_CLUSTERS)]

palette_data = pandas.DataFrame(columns = df_cols_color)
prop_data = pandas.DataFrame(columns = df_cols_prop)

#time code
import time
start_time = time.time()

# img_num = 2

feature_vectors = []

km = KMeans(n_clusters = 5,init='k-means++')

for img_num in range(400):
    # open image
    img = Image.open("./Kimono_Images/image_"+ str(img_num) +".jpg")
    rgb_array = np.array(img)
    rows,cols,p = rgb_array.shape

    # express image as single vector of pixels
    # CUT OFF HANDS AND PLINTH
    pixel_list = []
    for row in range(int(rows*0.2),int(rows*0.8)):
        for col in range(int(cols * 0.2), int(cols * 0.8)):
            if random.random() >0.90:
                pixel_list.append(srgb_to_oklab(rgb_array[row][col]))
                # print(rgb_array[row][col])

    # # feature vect based on km clustering + chroma-boost ---------------------
    # df = pd.DataFrame(pixel_list, columns = ['L','a','b'])
    # df = df[df['L'] <0.999]
    #
    # km.fit_predict(df)
    # print(km.cluster_centers_.tolist())

    # df['label'] = km.labels_
    # df['chroma'] = np.sqrt(df['a'] ** 2 + df['b'] ** 2)
    # most_chromatic_pixels = df.loc[df.groupby('label')['chroma'].idxmax()][['L', 'a', 'b']].values
    # chromatic_centroids = most_chromatic_pixels.tolist()
    # print(chromatic_centroids)
    # # + chromatic_centroids
    # feature_vector = km.cluster_centers_.tolist()
    # feature_vector = sorted(feature_vector)
    # feature_vector = np.array(feature_vector).flatten()
    # print(feature_vector)
    # feature_vectors.append(feature_vector)

    # # Initialize 3D counting matrix METHOD ---------------------------------------------
    counting_matrix = np.zeros((EMBED_DIM, EMBED_DIM, EMBED_DIM), dtype=int)
    bin_size = 1.0 / EMBED_DIM
    for l, a, b in pixel_list:
        if l > 0.999: continue # skip pure white!
        # Shift a and b into [0, 1] range
        a_shifted = a + 0.5
        b_shifted = b + 0.5

        # Compute bin indices
        l_idx = min(int(l // bin_size), EMBED_DIM - 1)
        a_idx = min(int(a_shifted // bin_size), EMBED_DIM - 1)
        b_idx = min(int(b_shifted // bin_size), EMBED_DIM - 1)

        counting_matrix[l_idx][a_idx][b_idx] += 1
        # counting_matrix[l_idx][a_idx][b_idx] = 1
    feature_vector = counting_matrix.reshape(1, -1)[0]
    feature_vectors.append(feature_vector)
    #
    if img_num % 100 == 0: print(img_num,"done")
print("FEATURE VECTORS GENERATED!")
# print(feature_vectors)
# print(shape(feature_vectors))
# print(feature_vectors)

data = pd.DataFrame(feature_vectors)
print(data.head())
data.to_csv(f"clustering_{TRIAL}_features.csv")
#
# feature_vectors = pd.read_csv(f"clustering_{TRIAL}_features.csv").iloc[:, 1:].to_numpy()
#
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)
print(feature_vectors)
from sklearn.decomposition import PCA
#### Reduce dimensionality from 512 to 50 components
pca = PCA(n_components=50)
feature_vectors_pca = pca.fit_transform(feature_vectors)


# K_MAX = 2000
# K_STEP = 10
# K_MIN = 50
# from sklearn.metrics import silhouette_score
# sil = []
# max_score = 0
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# for k in range(K_MIN, K_MAX+1,K_STEP):
#   kmeans = KMeans(n_clusters = k,init='k-means++').fit(feature_vectors)
#   labels = kmeans.labels_
#   current_score = silhouette_score(feature_vectors, labels, metric = 'euclidean')
#   sil.append(current_score)
#   print(f"k={k} score: {current_score}")
#   if current_score > max_score:
#       max_k = k
# print(f"max k:{max_k}")

km = KMeans(n_clusters=N_CLUSTERS, init='k-means++')
km.fit(feature_vectors)

clusters = pca.inverse_transform(km.cluster_centers_)
clusters = scaler.inverse_transform(clusters)
# clusters = km.cluster_centers_

np.set_printoptions(suppress=True, precision=2)

# print(clusters[0].reshape(EMBED_DIM, EMBED_DIM, EMBED_DIM))
print(km.labels_)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

# clusters = clusters*100
# clusters = clusters.astype(int)

print("Saving data")
data = pd.DataFrame(clusters)
print(data.head())
data.to_csv(f"clusters_{TRIAL}.csv")

cluster_map = defaultdict(list)
for idx, label in enumerate(km.labels_):
    cluster_map[label].append(idx)

# Save to a .txt file
with open(f"clusters_{TRIAL}_indices.txt", "w") as f:
    for cluster_id in sorted(cluster_map.keys()):
        indices = ", ".join(map(str, cluster_map[cluster_id]))
        f.write(f"{cluster_id}:\t{indices}\n")
