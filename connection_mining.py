import colorsys
import itertools
import math
import random
# from oklch_conv import rgb_direct_oklch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RAND_CUTOFF = 0.00

# convert from RGB (0-255, 0-255, 0-255) to a single number [0,512)
# or convert the other way around

DIMENSION = 8

def discretize_512(red, green, blue):
    r = int(red) // 32
    g = int(green) // 32
    b = int(blue) // 32
    return r*64 + g*8 + b
def undiscretize_512(num):
    red = num // 64
    green = (num // 8) % 8
    blue = num % 8
    return [red*32 + 16, green*32 + 16, blue*32 + 16]

# print(discretize_512(128,255,255))

for i in range(512):
    r,g,b = undiscretize_512(i)
    color = r*65536 + g*256 + b
    print( f"#{color:06X}")
# print(undiscretize_512(discretize_512(232,255,255)))

connection_matrix = np.zeros((512, 512), dtype=np.int64)

for img_num in range(4500):
    if img_num % 100 == 0:
        print("processing", img_num, end='... ')

    img = Image.open(f"./Kimono_Images/image_{img_num}.jpg")
    rgb_array = np.array(img)
    rows, cols, _ = rgb_array.shape

    # Crop
    sub = rgb_array[int(rows * 0.2):int(rows * 0.8), int(cols * 0.2):int(cols * 0.8)]
    # pixels = sub.reshape(-1, 3)
    pixels = sub.reshape(-1, 3).astype(np.int32)

    # Random cutoff mask
    mask = np.random.rand(len(pixels)) > RAND_CUTOFF
    pixels = pixels[mask]

    if len(pixels) == 0:
        continue

    # Discretize in vectorized form
    r = pixels[:, 0] // 32
    g = pixels[:, 1] // 32
    b = pixels[:, 2] // 32
    discr = (r * 64) + (g * 8) + b
    # print(discr)

    # Tally with bincount
    tally = np.bincount(discr, minlength=512)

    # Update connection matrix via outer product
    outer = np.outer(tally, tally)
    # connection_matrix += np.triu(outer, k=1)
    connection_matrix = outer

    if img_num % 100 == 0:
        print("done")

print(connection_matrix)
import csv

with open("connection_matrix_512.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(connection_matrix)
