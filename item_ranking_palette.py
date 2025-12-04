import pandas as pd
import numpy as np
from openai import OpenAI

CSV_PATH = "item_item_similarity_weighted_dim32.csv"   # change if needed

def load_similarity(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

SIM = load_similarity(CSV_PATH)

def generate_random_palette(sim_matrix, palette_size=5, top_k=20):
    """
    uses a random walk to create a color palette
    """
    all_colors = sim_matrix.index.tolist()

    # Start with a random color
    current_color = np.random.choice(all_colors)
    palette = [current_color]

    for _ in range(palette_size - 1):
        if current_color not in sim_matrix.index:
            break

        # Get top-K most similar colors
        sims = sim_matrix.loc[current_color].drop(current_color).nlargest(top_k)
        top_colors = sims.index.tolist()

        if not top_colors:
            break

        # Randomly pick the next color
        next_color = np.random.choice(top_colors)
        palette.append(next_color)
        current_color = next_color

    return palette
def generate_balanced_palette(sim_matrix, starting_color=None, palette_size=5, top_k=20):
    """
    attempts to maximize the next color's affinity with previous colors generated.
    Next color is chosen from top k affinities from the product of all previous affinities
    """

    all_colors = sim_matrix.index.tolist()

    # Start with a random color
    if starting_color is None:
        current_color = np.random.choice(all_colors)
        palette = [current_color]
    else:
        palette = [starting_color]

    for _ in range(palette_size - 1):
        # Compute cumulative score for all candidates
        scores = np.zeros(len(sim_matrix))
        for c in palette:
            scores += sim_matrix.loc[c].values

        candidates = np.array(all_colors)

        # Exclude already chosen colors
        mask = np.isin(candidates, palette, invert=True)
        filtered_candidates = candidates[mask]
        filtered_scores = scores[mask]

        if len(filtered_candidates) == 0:
            break

        # Take top-K highest scoring candidates
        top_idx = np.argsort(filtered_scores)[-top_k:]
        top_candidates = filtered_candidates[top_idx]

        # Randomly pick one from top-K
        next_color = np.random.choice(top_candidates)
        palette.append(next_color)

    return palette
def generate_very_balanced_palette(sim_matrix, starting_color=None, palette_size=5, top_k=20):

    all_colors = sim_matrix.index.tolist()

    # Start with a random color
    if starting_color is None:
        current_color = np.random.choice(all_colors)
        palette = [current_color]
    else:
        palette = [starting_color]

    for _ in range(palette_size - 1):
        print(top_k)
        # Compute cumulative score for all candidates
        scores = np.ones(len(sim_matrix))
        for c in palette:
            scores *= sim_matrix.loc[c].values

        candidates = np.array(all_colors)

        # Exclude already chosen colors
        mask = np.isin(candidates, palette, invert=True)
        filtered_candidates = candidates[mask]
        filtered_scores = scores[mask]

        if len(filtered_candidates) == 0:
            break

        # Take top-K highest scoring candidates
        top_idx = np.argsort(filtered_scores)[-top_k:]
        top_candidates = filtered_candidates[top_idx]

        # Randomly pick one from top-K
        next_color = np.random.choice(top_candidates)
        palette.append(next_color)

        top_k = int(top_k / 1.25) + 10

        client = OpenAI(api_key='replace')

        prompt = f"""
                Evaluate the color palette for a kimono:
                {palette}
                Give a 1-10 aesthetic rating and notes on the colory harmony
                """

        response = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": prompt}])

        print(response.choices[0].message.content)

    return palette

import matplotlib.pyplot as plt
def show_palette(palette, title=None):
    n = len(palette)
    fig, ax = plt.subplots(figsize=(n * 1.2, 1.5))

    for i, color in enumerate(palette):
        ax.fill_between([i, i + 1], 0, 1, color=color)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis('off')

    if title:
        ax.set_title(title)

    # Show the figure in a new window
    plt.show()

for i in range(10):
    palette = generate_very_balanced_palette(SIM,
                                        # starting_color='#EEFD50',
                                        palette_size=10, top_k=240)
    print("Generated palette:", palette)
    show_palette(palette,f"{i}")






