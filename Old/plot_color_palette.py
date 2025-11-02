import colorsys

import pandas as pd
import math
import random
from oklch_conv import rgb_direct_oklch
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image

NUM_KIMONOS = 4500
METHOD = "HSV"
N_K = 10
SCATTER_SIZE_SCALE = 10000

df = pd.read_csv("palette_data.csv")


for row in df.head(NUM_KIMONOS).itertuples(index=False):
    row_arr = np.array(row)
    img_num = row[0]
    row_arr = row_arr[1:]

    hue_angles = []
    radii = []
    scatter_colors = []
    scatter_sizes = []

    for color_i in range(N_K):
        r = row_arr[color_i * 3 + 0]/255
        g = row_arr[color_i * 3 + 1]/255
        b = row_arr[color_i * 3 + 2]/255
        scatter_colors.append([r,g,b])
        scatter_sizes.append(row_arr[N_K*3 + color_i]*SCATTER_SIZE_SCALE)
        # print((N_K*3 + color_i)*SCATTER_SIZE_SCALE)

        if METHOD == "HSV":
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hue_angles.append(h * 2 * math.pi)
            radii.append(math.sin(v * math.pi) * s)
        else:
            L, c, h = rgb_direct_oklch(rgb_array[row][col])
            # h is in degrees
            hue_angles.append(h * 2 * math.pi / 360 - 0.5)
            radii.append(c)

    hue_angles = np.array(hue_angles)
    radii = np.array(radii)

    # Create figure with polar projection
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    # customize
    fig.patch.set_facecolor('#808080')
    ax.set_facecolor('#808080')
    # ax.set_xticks(np.radians(np.arange(0, 360, 30)))  # Hue labels every 30 degrees
    # ax.set_xticklabels([f"{i}°" for i in range(0, 360, 30)])  # Label angles
    ax.set_yticklabels([])  # Remove radial labels
    if METHOD == "HSV": ax.set_ylim(0, 1)
    else: ax.set_ylim(0, 0.3)

    ax.grid(False)  # Remove grid

    ax.scatter(hue_angles, radii, c=scatter_colors, s=scatter_sizes, edgecolors='None')

    plt.savefig(f'plots_{METHOD}_palettes/plot_{img_num}.jpg', format='jpg', dpi=300)  # Save as JPG with 300 dpi
    # plt.show()
    plt.close()

    if img_num%50 == 0: print(img_num,"done")


