import colorsys
import math
import random
# from oklch_conv import rgb_direct_oklch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAND_CUTOFF = 0.97
METHOD = "HSV"  # or "OKLch"

hue_angles = []
radii = []
scatter_colors = []

for img_num in range(4500):
    if img_num%100 == 0: print("processing", img_num, end='... ')
    img = Image.open(f"./Kimono_Images/image_{img_num}.jpg")
    rgb_array = np.array(img)
    rows, cols, _ = rgb_array.shape

    for row in range(int(rows * 0.2), int(rows * 0.8)):
        for col in range(int(cols * 0.2), int(cols * 0.8)):
            if random.random() > RAND_CUTOFF:
                r, g, b = rgb_array[row][col]
                r, g, b = r / 255.0, g / 255.0, b / 255.0
                scatter_colors.append([r, g, b])

                if METHOD == "HSV":
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hue_angles.append(h * 2 * math.pi)
                    radii.append(math.sin(v * math.pi) * s)
                # elif METHOD == "OKLch":
                #     L, c, h = rgb_direct_oklch(rgb_array[row][col])
                #     hue_angles.append(h * 2 * math.pi / 360 - 0.5)
                #     radii.append(c)
    if img_num%100 == 0: print("done")

# Convert to arrays
hue_angles = np.array(hue_angles)
radii = np.array(radii)
scatter_colors = np.array(scatter_colors)

# Create polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
fig.patch.set_facecolor('#808080')
ax.set_facecolor('#808080')
ax.set_yticklabels([])
ax.grid(False)
if METHOD == "OKLch":
    ax.set_ylim(0, 0.3)
else:
    ax.set_ylim(0, 1.1)

# Scatter with low alpha
ax.scatter(hue_angles, radii, c=scatter_colors, s=30, alpha=0.003, edgecolors='none')

plt.savefig(f'combined_plot_{METHOD}.jpg', format='jpg', dpi=300)
plt.close()
print("Saved combined plot.")
