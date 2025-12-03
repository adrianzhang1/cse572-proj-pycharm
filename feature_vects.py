import numpy as np
from PIL import Image
from Old.oklab_conv import srgb_to_oklab, oklab_to_srgb

DIMENSION = 32
RAND_CUTOFF = 0.8
NUM_BUCKETS = DIMENSION ** 3
N_IMAGES = 4500

def discretize_lab(labs, dim):
    bin_size = 1.0 / dim

    L = labs[:, 0]
    a = labs[:, 1] + 0.5
    b = labs[:, 2] + 0.5

    L_idx = np.floor(L / bin_size).astype(int)
    a_idx = np.floor(a / bin_size).astype(int)
    b_idx = np.floor(b / bin_size).astype(int)

    flat_idx = L_idx * (dim ** 2) + a_idx * dim + b_idx
    return flat_idx
def discretized_lab_to_hex(idx, dim):
    # Recover (L_idx, a_idx, b_idx)
    L_idx = idx // (dim * dim)
    a_idx = (idx // dim) % dim
    b_idx = idx % dim

    bin_size = 1.0 / dim

    # Reconstruct OKLab values at the CENTER of the bucket
    L = (L_idx + 0.5) * bin_size
    a = (a_idx + 0.5) * bin_size - 0.5
    b = (b_idx + 0.5) * bin_size - 0.5

    # Convert back to sRGB
    r,g,b = oklab_to_srgb([L, a, b])  # result should be float in [0,1]

    # # Clamp (OKLab→sRGB sometimes goes slightly outside [0,1])
    # rgb = np.clip(rgb, 0.0, 1.0)
    #
    # # Convert to hex
    # r = int(rgb[0] * 255)
    # g = int(rgb[1] * 255)
    # v = int(rgb[2] * 255)

    return f"#{r:02X}{g:02X}{b:02X}"


all_histograms = []
nonzero_global = np.zeros(NUM_BUCKETS, dtype=bool)

for img_num in range(N_IMAGES):
    if img_num % 50 == 0:
        print("Processing image", img_num)

    img = Image.open(f"./Kimono_Images/image_{img_num}.jpg")
    rgb = np.array(img)

    H, W, _ = rgb.shape
    sub = rgb[int(H*0.2):int(H*0.8), int(W*0.2):int(W*0.8)]
    pixels = sub.reshape(-1, 3)

    if RAND_CUTOFF > 0:
        mask = np.random.rand(len(pixels)) > RAND_CUTOFF
        pixels = pixels[mask]

    if len(pixels) == 0:
        hist = np.zeros(NUM_BUCKETS, dtype=np.int32)
        all_histograms.append(hist)
        continue

    labs = np.array([srgb_to_oklab(p) for p in pixels])
    idx = discretize_lab(labs, DIMENSION)

    hist = np.bincount(idx, minlength=NUM_BUCKETS).astype(np.int32)
    all_histograms.append(hist)

    # Track which LAB bins ever appear
    nonzero_global |= (hist > 0)

print("Pass 1 complete.")
used_bins = np.where(nonzero_global)[0]
print(f"Total non-zero global bins: {len(used_bins)}")

out_file = f"kimono_LAB_histograms_dim{DIMENSION}_compact.csv"

with open(out_file, "w") as f:
    # Header = only used bins
    header_hex = [discretized_lab_to_hex(i, DIMENSION) for i in used_bins]
    f.write(",".join(header_hex) + "\n")

    # Write each histogram using only the non-zero bins
    for hist in all_histograms:
        row = hist[used_bins]
        f.write(",".join(map(str, row)) + "\n")

print("Done. Output saved to", out_file)
