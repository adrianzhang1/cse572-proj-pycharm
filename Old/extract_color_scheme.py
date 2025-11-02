# NOTE: some code referenced from
# https://www.generativelabs.co/post/using-k-means-clustering-to-create-a-color-palette-from-reference-images
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

N_CLUSTERS = 10
GENERATE_VISUALS = True
km = KMeans(n_clusters = N_CLUSTERS, init = 'k-means++')

df_cols_color = [f'color_{i + 1}_{ch}' for i in range(N_CLUSTERS) for ch in ['L', 'a', 'b']]
df_cols_prop = [f'p_{i + 1}' for i in range(N_CLUSTERS)]

palette_data = pandas.DataFrame(columns = df_cols_color)
prop_data = pandas.DataFrame(columns = df_cols_prop)

#time code
import time
start_time = time.time()

# img_num = 2
for img_num in range(800):
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
            pixel_list.append(srgb_to_oklab(rgb_array[row][col]))
        # pixel_list.extend(rgb_array[row][int(cols*0.2):int(cols*0.8)])

    # cap white values at 254 to remove pure white background
    # df = pd.DataFrame(pixel_list, columns = ['R','G','B'])
    # df = df[(df['R'] < 254) & (df['G'] < 254) & (df['B'] < 254)]
    df = pd.DataFrame(pixel_list, columns = ['L','a','b'])
    df = df[df['L'] <0.999]
    # df = df.drop_duplicates()
    num_pixels = len(df)
    # print(pixel_list)
    # print(df.head())

    # Compute kmeans
    X = df
    y = km.fit_predict(X)

    # centroids = km.cluster_centers_.tolist()
    # centroids = [[math.floor(c[0]),math.floor(c[1]),math.floor(c[2])] for c in centroids]
    # centroids = [[int(c[0]), int(c[1]), int(c[2])] for c in centroids]
    # print(centroids)
    labels = km.labels_
    # print(labels)

    df['label'] = labels
    df['chroma'] = np.sqrt(df['a'] ** 2 + df['b'] ** 2)

    # Pick the pixel with the highest chroma per cluster
    most_chromatic_pixels = df.loc[df.groupby('label')['chroma'].idxmax()][['L', 'a', 'b']].values
    centroids = most_chromatic_pixels.tolist()

    proportions = np.bincount(y)
    proportions = [p / num_pixels for p in proportions]

    # print(proportions)

    # GENERATE VISUALIZER IMAGE
    if GENERATE_VISUALS:
        np_image = []
        counter = 0
        for i,num in enumerate(proportions):
            # print(i,num)
            for j in range(int(num*200)):
                np_image.append([oklab_to_srgb(centroids[i])]*200)

        rgb_array = np.array(np_image, dtype=np.uint8)
        visual = Image.fromarray(rgb_array)
        # visual.show()
        visual.save("./visuals-new-strat-"+str(N_CLUSTERS)+"/palette"+ str(img_num) +".png")

    # save into DF
    palette_row = [channel for color in centroids for channel in color]
    # print(centroids)
    # print(proportions)
    # print(row_data)
    palette_data.loc[len(palette_data)] = palette_row
    prop_data.loc[len(prop_data)] = proportions

    if img_num%25 == 0:
        print(img_num,"done")
        # print("Saving data")
        # data = pd.concat([palette_data, prop_data], axis=1)
        # data.to_csv("palette_data_OKLAB_"+str(N_CLUSTERS)+"_clusters.csv")


end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

# print("Saving data")
# data = pd.concat([palette_data, prop_data], axis=1)
# print(data.head())
# data.to_csv("palette_data_OKLAB_"+str(N_CLUSTERS)+"_clusters.csv")
