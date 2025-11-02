# NOTE: some code referenced from
# https://www.generativelabs.co/post/using-k-means-clustering-to-create-a-color-palette-from-reference-images

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from PIL import Image

img_num = 1

img = Image.open("./Kimono_Images/image_"+ str(img_num) +".jpg")
# img = Image.open("twitter_images-nov2024/image-" + str(img_num+1) + ".jpg")
rgb_array = np.array(img)
rows,cols,p = rgb_array.shape

# express image as single vector of pixels
# CUT OFF HANDS AND PLINTH
pixel_list = []
# for row in rgb_array:
#     pixel_list.extend(row)
for row in range(int(rows*0.2),int(rows*0.8)):
    for col in range(int(cols * 0.2), int(cols * 0.8)):
        pixel_list.append(rgb_array[row][col])
    # pixel_list.extend(rgb_array[row][int(cols*0.2):int(cols*0.8)])

# cap white values at 254 to remove pure white background
df = pd.DataFrame(pixel_list, columns = ['R','G','B'])
df = df[(df['R'] < 254) & (df['G'] < 254) & (df['B'] < 254)]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Create and visualize the dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(df_scaled, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Perform clustering without specifying n_clusters
hc = AgglomerativeClustering(distance_threshold=3, n_clusters=None, linkage="ward")
clusters = hc.fit_predict(df_scaled)

# Add the cluster labels to the DataFrame
df["Cluster"] = clusters
print(df)
















from math import floor
from oklch_conv import rgb_direct_oklch

# Load image and display
# img_original = cv2.imread(image_path)
# print('Dimensions : ',img_original.shape)
# print(img)
import math

import pandas
from PIL import Image
import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
import sklearn.cluster.hierarchy as sch

N_CLUSTERS = 10
GENERATE_VISUALS = True
# km = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++')

df_cols_color = [f'color_{i + 1}_{ch}' for i in range(N_CLUSTERS) for ch in ['R', 'G', 'B']]
df_cols_prop = [f'p_{i + 1}' for i in range(N_CLUSTERS)]

palette_data = pandas.DataFrame(columns = df_cols_color)
prop_data = pandas.DataFrame(columns = df_cols_prop)

#time code
import time
start_time = time.time()

# img_num = 2
for img_num in range(1):
    # open image
    img = Image.open("./Kimono_Images/image_"+ str(img_num) +".jpg")
    # img = Image.open("twitter_images-nov2024/image-" + str(img_num+1) + ".jpg")
    rgb_array = np.array(img)
    rows,cols,p = rgb_array.shape

    # express image as single vector of pixels
    # CUT OFF HANDS AND PLINTH
    pixel_list = []
    # for row in rgb_array:
    #     pixel_list.extend(row)
    for row in range(int(rows*0.2),int(rows*0.8)):
        for col in range(int(cols * 0.2), int(cols * 0.8)):
            pixel_list.append(rgb_direct_oklch(rgb_array[row][col]))
        # pixel_list.extend(rgb_array[row][int(cols*0.2):int(cols*0.8)])

    # cap white values at 254 to remove pure white background
    df = pd.DataFrame(pixel_list, columns = ['R','G','B'])
    df = df[(df['R'] < 254) & (df['G'] < 254) & (df['B'] < 254)]
    # df = df.drop_duplicates()
    num_pixels = len(df)

    # Compute kmeans
    # X = df
    # y = km.fit_predict(X)
    dendrogram = sch.dendrogram(sch.linkage(df, method="ward"))

    # centroids = km.cluster_centers_.tolist()
    # # centroids = [[math.floor(c[0]),math.floor(c[1]),math.floor(c[2])] for c in centroids]
    # centroids = [[int(c[0]), int(c[1]), int(c[2])] for c in centroids]
    # # print(centroids)
    #
    # proportions = np.bincount(y)
    # proportions = [p / num_pixels for p in proportions]
    #
    # # print(proportions)
    #
    # # GENERATE VISUALIZER IMAGE
    # if GENERATE_VISUALS:
    #     np_image = []
    #     counter = 0
    #     for i,num in enumerate(proportions):
    #         # print(i,num)
    #         for j in range(int(num*200)):
    #             np_image.append([(centroids[i])]*200)
    #
    #     rgb_array = np.array(np_image, dtype=np.uint8)
    #     visual = Image.fromarray(rgb_array)
    #     # visual.show()
    #     visual.save("./visuals-OKLAB-no-hands"+str(N_CLUSTERS)+"/palette"+ str(img_num) +".png")
    #
    # # save into DF
    # palette_row = [channel for color in centroids for channel in color]
    # # print(centroids)
    # # print(proportions)
    # # print(row_data)
    # palette_data.loc[len(palette_data)] = palette_row
    # prop_data.loc[len(prop_data)] = proportions
    #
    # if img_num%25 == 0:
    #     print(img_num,"done")
    #     print("Saving data")
    #     data = pd.concat([palette_data, prop_data], axis=1)
    #     data.to_csv("twit_palette_data_" + str(N_CLUSTERS) + "_color.csv")


end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

print("Saving data")
data = pd.concat([palette_data, prop_data], axis=1)
print(data.head())
data.to_csv("palette_data_OKLCH_"+str(N_CLUSTERS)+"_clusters.csv")





