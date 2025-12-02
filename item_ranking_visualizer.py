import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

CSV_PATH = "item_item_similarity_dim32.csv"   # change if needed


def load_similarity(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


try:
    SIM = load_similarity(CSV_PATH)
except Exception as e:
    print("Error loading CSV:")
    print(e)
    raise


def plot_top_similar(color, k=20):
    if color not in SIM.index:
        messagebox.showerror("Error", f"Color '{color}' not found.")
        return

    sims = SIM.loc[color].drop(color, errors="ignore").nlargest(k).sort_values()
    colors = sims.index.tolist()
    values = sims.values
    n = len(colors)

    plt.figure(figsize=(10, max(2, n * 0.35)))
    plt.barh(range(n), values, color=colors, edgecolor='black', linewidth=0.3)
    plt.yticks(range(n), colors, fontsize=9)
    plt.xlabel("Cosine similarity")
    plt.title(f"Top {n} similar colors to {color}")
    plt.tight_layout()
    plt.show()


def show_color_grid(color, k=20, cols=5):
    if color not in SIM.index:
        messagebox.showerror("Error", f"Color '{color}' not found.")
        return

    sims = SIM.loc[color].drop(color).nlargest(k)
    colors = sims.index.tolist()
    values = sims.values

    rows = int(np.ceil(len(colors) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.3))
    axs = axs.flatten()

    for ax in axs[len(colors):]:
        ax.axis('off')

    for i, (hex_color, sim_val) in enumerate(zip(colors, values)):
        ax = axs[i]
        ax.set_facecolor(hex_color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.5, 0.5,
            f"{hex_color}\n{sim_val:.3f}",
            ha='center', va='center',
            color='white' if sim_val > 0.3 else 'black',
            fontsize=7
        )

        ax.patch.set_edgecolor("black")
        ax.patch.set_linewidth(0.5)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Build GUI
# ----------------------------------------------------------

root = tk.Tk()
root.title("Color Similarity Explorer")
root.geometry("500x200")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

# Dropdown
ttk.Label(frame, text="Pick a color:").grid(row=0, column=0, sticky="w")

all_colors = sorted(SIM.index.tolist())
color_var = tk.StringVar(value=all_colors[0])

dropdown = ttk.Combobox(frame, textvariable=color_var, values=all_colors, width=20)
dropdown.grid(row=0, column=1, sticky="w")

# ----------------------------------------------------------
# Add color preview swatch
# ----------------------------------------------------------

preview_label = tk.Label(frame, text="       ", relief="solid", borderwidth=1)
preview_label.grid(row=0, column=2, padx=15)

def update_preview(*args):
    color = color_var.get()
    try:
        preview_label.config(background=color)
    except:
        preview_label.config(background="white")

color_var.trace_add("write", update_preview)
update_preview()

# K selector
ttk.Label(frame, text="Top K:").grid(row=1, column=0, sticky="w")
k_var = tk.IntVar(value=20)
k_entry = ttk.Entry(frame, textvariable=k_var, width=5)
k_entry.grid(row=1, column=1, sticky="w")


def on_plot():
    plot_top_similar(color_var.get(), k_var.get())


def on_grid():
    show_color_grid(color_var.get(), k_var.get())


btn1 = ttk.Button(frame, text="Show Bar Chart", command=on_plot)
btn1.grid(row=2, column=0, pady=10)

btn2 = ttk.Button(frame, text="Show Color Grid", command=on_grid)
btn2.grid(row=2, column=1, pady=10)

root.mainloop()
