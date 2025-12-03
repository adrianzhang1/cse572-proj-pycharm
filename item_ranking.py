import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("kimono_LAB_histograms_dim32_compact.csv")

# df shape: (num_kimonos, num_bins_kept)

df_norm = df.div(df.sum(axis=1), axis=0).fillna(0)

item_vectors = df_norm.values.T  # shape: (num_bins, num_kimonos)
item_sim = cosine_similarity(item_vectors)  # (num_bins × num_bins)

item_labels = df_norm.columns.tolist()
item_sim_df = pd.DataFrame(item_sim, index=item_labels, columns=item_labels)

# item_sim_df.to_csv("item_item_similarity_dim32.csv")
# print("Saved item_item_similarity_dim32.csv")

# item_sim_df = pd.read_csv("item_item_similarity_dim32.csv")
# item_labels = item_sim_df.columns.tolist()

print(item_sim_df.head())

support_counts = df.sum(axis=0)  # Series, index = hex bins
# Apply support weighting
weight_matrix = np.sqrt(np.outer(support_counts, support_counts))
weighted_sim = item_sim_df.values * weight_matrix

item_sim_df_weighted = pd.DataFrame(weighted_sim, index=df.columns, columns=df.columns)
item_sim_df_weighted.to_csv("item_item_similarity_weighted_dim32.csv")
def top_similar_colors(hex_color, k=10):
    if hex_color not in item_sim_df.index:
        raise ValueError(f"Color {hex_color} not found in data.")

    row = item_sim_df.loc[hex_color]
    return row.drop(hex_color).nlargest(k)

if __name__ == "__main__":
    # Pick any LAB-bin hex in your dataset
    example_hex = item_labels[100]  # or put "#8F4A33" or any known value

    print(f"\nTop similar colors to {example_hex}:")
    print(top_similar_colors(example_hex, k=10))
