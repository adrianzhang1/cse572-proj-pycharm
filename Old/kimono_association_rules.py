from collections import defaultdict
from math import floor

import pandas as pd

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

EMBED_DIM = 8
TRIAL = "0.05"

#time code
import time
start_time = time.time()

feature_vectors = []

km = KMeans(n_clusters = 5,init='k-means++')

for img_num in range(4500):
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

    # # Initialize 3D counting matrix METHOD ---------------------------------------------
    counting_matrix = np.zeros((EMBED_DIM, EMBED_DIM, EMBED_DIM), dtype=int)
    bin_size = 1.0 / EMBED_DIM
    total_count = 0
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
        total_count += 1
        # counting_matrix[l_idx][a_idx][b_idx] = 1
    for l in range(EMBED_DIM):
        for a in range(EMBED_DIM):
            for b in range(EMBED_DIM):
                # print(l,a,b)
                if counting_matrix[l,a,b] > 4:
                    count = counting_matrix[l][a][b]
                    lx = (l + 0.5) / EMBED_DIM
                    ax = (a + 0.5) / EMBED_DIM - 0.5
                    bx = (b + 0.5) / EMBED_DIM - 0.5
                    rgb = oklab_to_srgb([lx,ax,bx])
                    # print(rgb)
    # print(total_count)

    # feature_vector = counting_matrix.reshape(1, -1)[0]
    # feature_vectors.append(feature_vector)
    # Normalize matrix to get frequencies
    freq_matrix = counting_matrix / total_count

    # Thresholds
    one_hot_low = (freq_matrix >= 0.05).astype(int)
    # one_hot_high = (freq_matrix >= 100 / 1000).astype(int)
    #
    # # Flatten both into vectors
    # one_hot_low_vector = one_hot_low.flatten()
    # one_hot_high_vector = one_hot_high.flatten()
    #
    # # Combine them if desired (e.g., as 2*512 vector or DataFrame with separate sets)
    # combined_vector = np.concatenate([one_hot_low_vector, one_hot_high_vector])

    # df_row = pd.Series(combined_vector)

    df_row = pd.Series(one_hot_low.flatten())
    feature_vectors.append(df_row)
    if img_num % 100 == 0: print(img_num,"done")
print("FEATURE VECTORS GENERATED!")

df = pd.DataFrame(feature_vectors)

# create descruptive labels
labels = []
# for threshold in ['low', 'high']:  # corresponds to 1/1000 and 100/1000
for l in range(EMBED_DIM):
    for a in range(EMBED_DIM):
        for b in range(EMBED_DIM):
            # Compute bin center in OKLab
            lx = (l + 0.5) / EMBED_DIM
            ax = (a + 0.5) / EMBED_DIM - 0.5
            bx = (b + 0.5) / EMBED_DIM - 0.5

            # Convert to RGB
            rgb = oklab_to_srgb([lx, ax, bx])
            r, g, b_val = [round(x, 3) for x in rgb]

            # Construct label
            label = f'rgb({r},{g},{b_val})'
            labels.append(label)
# print(labels)

df = df.astype(bool)
df.columns = labels

print(df.describe())
print(df.head())

df.to_csv(f"kimono_association_itemsets_{TRIAL}.csv")
print("SAVED!")

# df = pd.read_csv("kimono_rule_itemsets.csv")
# df = df.astype(bool)
# df = df.loc[:, ~df.columns.str.startswith('high')]
# df = df.drop('Unnamed: 0', axis=1)

from mlxtend.frequent_patterns import fpgrowth, association_rules

# Minimum support — e.g., appears in at least 5% of the images
frequent_itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)

# Display some frequent patterns
print(frequent_itemsets.sort_values('support', ascending=False).head())

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Clean view
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head()

rules.to_csv(f"kimono_association_rules_{TRIAL}.csv")

print(rules)