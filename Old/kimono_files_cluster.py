import os
import shutil


# Path to the file containing clusters
cluster_file_path = "kimono_clusters_183_indices.txt"  # Replace with your actual filename
source_dir = "./Kimono_Images/"
target_root = "./clustered_kimonos_183/"

# Create the target root directory if it doesn't exist
os.makedirs(target_root, exist_ok=True)

import pandas as pd

# clusters = pd.read_csv("kimono_clusters_183 NONBINARY.csv")
# clusters = clusters.to_numpy()
# clusters = clusters[:,1:]
#
# print(clusters[12])
# count = 0

import matplotlib.pyplot as plt
import numpy as np
from oklab_conv import srgb_to_oklab,oklab_to_srgb
import colorsys

EMBED_DIM = 16

# import numpy as np
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


with open(cluster_file_path, 'r') as f:
    for line in f:
        # Parse each line in format: 0:\t29, 39, 54, ...
        cluster_id, indices_str = line.strip().split(":\t")
        # cluster_id = int(cluster_id)
        image_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]

        # Create directory for this cluster
        cluster_dir = os.path.join(target_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Copy each image
        for idx in image_indices:
            src = os.path.join(source_dir, f"image_{idx}.jpg")
            dst = os.path.join(cluster_dir, f"image_{idx}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {src} not found.")

# print(count)

