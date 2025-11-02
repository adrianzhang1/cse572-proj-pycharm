import os
import shutil


# Path to the file containing clusters
# cluster_file_path = "kimono_clusters_183_indices.txt"  # Replace with your actual filename
source_dir = "./Kimono_Images/"

# TRIAL = "dim32-k500"
TRIAL = "dim32-k300-nb"
# OUTPUT_TRIAL = "dim8-k300"
OUTPUT_TRIAL = TRIAL
# TRIAL = "k-10-hypercluster-normal"
# Create the target root directory if it doesn't exist
# os.makedirs(target_root, exist_ok=True)
# os.makedirs(f"./clustering_visuals_{TRIAL_NUM}", exist_ok=True)
os.makedirs(f"./clustering_visuals_{OUTPUT_TRIAL}", exist_ok=True)
cluster_file_path = f"clusters_{TRIAL}_indices.txt"
target_root = f"./clustered_kimonos_{OUTPUT_TRIAL}/"


import pandas as pd

# clusters = pd.read_csv("kimono_clusters_183 NONBINARY.csv")
clusters = pd.read_csv(f"clusters_{TRIAL}.csv")
clusters = clusters.to_numpy()
clusters = clusters[:,1:]

# print(clusters[12])
count = 0

import matplotlib.pyplot as plt
import numpy as np
from oklab_conv import *
import colorsys

EMBED_DIM = 32

# import numpy as np
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

from PIL import Image, ImageDraw

with open(cluster_file_path, 'r') as f:
    for line in f:
        # Parse each line in format: 0:\t29, 39, 54, ...
        cluster_id, indices_str = line.strip().split(":\t")
        cluster_id = int(cluster_id)
        image_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]

        # COPY IMAGES ----------------------
        # Create directory for this cluster
        cluster_dir = os.path.join(target_root, f"cluster_{cluster_id}_{len(image_indices)}pts")
        os.makedirs(cluster_dir, exist_ok=True)

        # Copy each image
        for idx in image_indices:
            src = os.path.join(source_dir, f"image_{idx}.jpg")
            dst = os.path.join(cluster_dir, f"image_{idx}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {src} not found.")

        print(len(image_indices))
        # # CREATE VISUALS FOR n^3 EMBEDDING SPACE ----------------------
        # if len(image_indices) < 2: continue
        # else: count += 1
        #
        # # Assume these are defined
        cluster = clusters[cluster_id]  # shape: (EMBED_DIM**3,)
        cluster_3d = cluster.reshape((EMBED_DIM, EMBED_DIM, EMBED_DIM))

        # Get non-zero bins
        indices = np.argwhere(cluster_3d > 0)
        counts = cluster_3d[cluster_3d > 0]

        # Convert bin indices to normalized [0, 1] OKLab coordinates
        bin_size = 1.0 / EMBED_DIM
        L_vals = (indices[:, 0] + 0.5) * bin_size
        a_vals = (indices[:, 1] + 0.5) * bin_size - 0.5  # shift back from [0,1] to [-0.5,0.5]
        b_vals = (indices[:, 2] + 0.5) * bin_size - 0.5

        # Convert to RGB colors
        rgb_colors = np.array([
            oklab_to_srgb([L, a, b])
            for L, a, b in zip(L_vals, a_vals, b_vals)
        ])
        lab_colors = np.array([
            # oklab_to_oklch([L, a, b])
            (L,a,b)
            for L, a, b in zip(L_vals, a_vals, b_vals)
        ])

        # # Clip RGB values to [0,1] to avoid matplotlib errors
        rgb_colors = rgb_colors/255
        # # print(rgb_colors)
        #
        # Normalize sizes
        # sizes = (counts / counts.max()) * 2000
        sizes = (counts / counts.max()) * 400
        #
        #
        # # Plot 3D
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # sc = ax.scatter(L_vals, a_vals, b_vals, c=rgb_colors, s=sizes,alpha=1.0)
        #
        # ax.set_xlabel("L")
        # ax.set_ylabel("a")
        # ax.set_zlabel("b")
        # ax.set_title(f"3D Cluster Visualization with OKLab Colors {cluster_id}")
        #
        # plt.tight_layout()
        # plt.show()

        ## FOR THE HYPERCLUSTERING
        # clusterLAB = cluster.reshape((10,3))
        # rgb_colors = ([oklab_to_srgb(LAB) for LAB in clusterLAB])
        # rgb_colors = [(r/255,g/255,b/255) for r,g,b in rgb_colors]
        # L_vals = [l for l,a,b in clusterLAB]
        # a_vals = [a for l, a, b in clusterLAB]
        # b_vals = [b for l, a, b in clusterLAB]
        # sizes = [300]*10

        hue_angles = []
        radii = []
        # HSV METHOD
        for r,g,b in rgb_colors:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hue_angles.append(h * 2 * math.pi)
            # radii.append(math.sin(v * math.pi) * s)
            # radii.append(s)
            # radii.append(min(s,abs(0.5-v)*2))
            radii.append(min(s,v))

        # #LAB METHOD
        DISTORT = 0.05
        slope = math.sqrt(2)
        for i,l in enumerate(L_vals):
            a_vals[i] += l * DISTORT * slope
            b_vals[i] += l * DISTORT

        # cluster_dir = os.path.join(target_root, f"cluster_{cluster_id}_{len(image_indices)}pts")
        # RADIAL PLOT!!
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        # customize
        fig.patch.set_facecolor('#808080')
        ax.set_facecolor('#808080')
        ax.set_yticklabels([])  # Remove radial labels
        ax.grid(False)  # Remove grid
        ax.set_ylim(0,1.05)
        ax.set_thetagrids(np.arange(0, 360, 15))
        ax.scatter(hue_angles, radii, c=rgb_colors, s=sizes, edgecolors='None')
        # plt.show()
        plt.savefig(f"./clustering_visuals_{OUTPUT_TRIAL}/HSV_cluster_{cluster_id}_{len(image_indices)}pts")
        plt.savefig(f"{target_root}cluster_{cluster_id}_{len(image_indices)}pts/0_HSV_vis.jpg")
        plt.close()

        # 2d SCATTER PLOT
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#808080')
        ax.set_facecolor('#808080')
        # ax.set_yticklabels([])  # Remove radial labels
        ax.grid(False)  # Remove grid
        ax.set_xlim(-0.2, 0.4)
        ax.set_ylim(-0.3, 0.3)
        ax.scatter(a_vals, b_vals, c=rgb_colors, s=sizes, edgecolors='None',clip_on=False,linewidths=0.5)
        # plt.show()
        plt.savefig(f"./clustering_visuals_{OUTPUT_TRIAL}/LAB_cluster_{cluster_id}_{len(image_indices)}pts")
        plt.savefig(f"{target_root}cluster_{cluster_id}_{len(image_indices)}pts/0_LAB_vis.jpg")
        # plt.savefig(os.path.join(cluster_dir, "0_HSV_cluster.jpg"))
        plt.close()

        # # CREATE palettes FOR 20 K-CLUSTER SPACE ----------------------
        # cluster = clusters[cluster_id]
        # clusterLAB = cluster.reshape((20,3))
        # clusterRGB = ([oklab_to_srgb(LAB) for LAB in clusterLAB])
        # # print(clusterRGB)
        #
        # swatch_width = 100
        # swatch_height = 20  # 20px per swatch, for 20 colors = 400px tall
        # img = Image.new("RGB", (swatch_width, swatch_height * len(clusterRGB)), color="white")
        # draw = ImageDraw.Draw(img)
        #
        # # Draw each color block
        # for i, color in enumerate(clusterRGB):
        #     top = i * swatch_height
        #     bottom = top + swatch_height
        #     draw.rectangle([0, top, swatch_width, bottom], fill=color)
        #
        # # Save or show
        # img.save(f"./clustering_visuals_{TRIAL}/cluster_{cluster_id}_{len(image_indices)}.png")
        # # img.show()



print(count)

