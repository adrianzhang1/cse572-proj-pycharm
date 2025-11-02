import colorsys
import math
import random
from oklch_conv import rgb_direct_oklch

# import color

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAND_CUTOFF = 0.97
METHOD = "HSV" # changes the

for img_num in range(100,400):
    print("working on",img_num,end='')
    img = Image.open(f"./Kimono_Images_Yumeya/image_{img_num}.png")
    # img = Image.open(f"test_all_colors.webp")
    rgb_array = np.array(img)
    rows,cols,p = rgb_array.shape

    hue_angles = []
    radii = []
    scatter_colors = []

    i = 0
    # cut out the hands, head, and plinth from the image
    # for row in range(rows):
    #     for col in range(cols):
    for row in range(int(rows*0.2),int(rows*0.8)):
        for col in range(int(cols*0.2),int(cols*0.8)):
            if random.random()>RAND_CUTOFF:
                r,g,b = rgb_array[row][col]
                r, g, b = r / 255.0, g / 255.0, b / 255.0
                scatter_colors.append([r, g, b])

                if METHOD == "HSV":
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hue_angles.append(h * 2 * math.pi)
                    radii.append(math.sin(v * math.pi)*s)

                elif METHOD == "OKLch":
                    L, c, h = rgb_direct_oklch(rgb_array[row][col])
                    # h is in degrees
                    hue_angles.append(h * 2 * math.pi/360-0.5)
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
    if METHOD=="OKLch": ax.set_ylim(0, 0.3)  # Limit radius to keep points within wheel
    else: ax.set_ylim(0, 1.1)
    ax.grid(False)  # Remove grid

    ax.scatter(hue_angles, radii, c=scatter_colors, s=30, edgecolors='None')

    plt.savefig(f'plots_{METHOD}/yumeya_plot_{img_num}.jpg', format='jpg', dpi=300)  # Save as JPG with 300 dpi
    # plt.savefig("TEST.jpg", format='jpg', dpi=300)
    # plt.show()

    plt.close()

    print(" done")

