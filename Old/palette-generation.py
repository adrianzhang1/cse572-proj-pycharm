import tkinter as tk
import random
from tkinter import colorchooser
from tkinter import ttk
from PIL import Image, ImageTk
import colorsys
import matplotlib.colors as mcolors

# switch image for matsuda type
def change_image(event):
    # selected_image = combo.get()
    # if selected_image in image_paths:
    #     img = Image.open(image_paths[selected_image])
    #     img = img.resize((128, 128), Image.LANCZOS)
    #     photo = ImageTk.PhotoImage(img)
    #     image_label.config(image=photo)
    #     image_label.image = photo
    pass


# change color selected
def change_color():
    color = colorchooser.askcolor(title="Select Color")
    # print(color_hex)
    # color_rgb = mcolors.to_rgb(color_hex)
    # color_hsv = c
    if color[1]:
        frame.config(bg=color[1])


# TODO: below
# generate palette based on matsuda type and given color

def hsv_to_hex(h,s,v):
    r, g, b  = colorsys.hsv_to_rgb(h,s,v)

    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
def gen_palette():
    bg_color = frame.cget("bg")
    matsuda_type = combo.get()

    print(bg_color)
    main_rgb = [color/255 for color in mcolors.to_rgb(bg_color)]
    main_hsv = colorsys.rgb_to_hsv(main_rgb[0],main_rgb[1],main_rgb[2])
    main_hue = main_hsv[0]
    # print(main_hsv)

    col1_disp.config(bg=bg_color)
    hues = []
    sats = [random.random() for _ in range(4)]
    vals = [random.random() for _ in range(4)]

    # for i in range(4):
    match matsuda_type:
        case "i Type":
            [hues.append((random.uniform(-1 / 24, 1 / 24) + main_hue) % 1) for _ in range(4)]
        case "I Type":
            [hues.append((random.uniform(-1 / 24, 1 / 24) + main_hue) % 1) for _ in range(2)]
            [hues.append((random.uniform(-1 / 24, 1 / 24) + main_hue + 0.5) % 1) for _ in range(2)]
            hues[0] = (main_hue+0.5) % 1
            sats[0],vals[0] = main_hsv[1],main_hsv[2]
            print(hues[0],sats[0],vals[0])
        case "V Type":
            [hues.append((random.uniform(-1 / 8, 1 / 8) + main_hue) % 1) for _ in range(4)]
        case _:
            hues = [random.random() for _ in range(4)]

    col2_disp.config(bg=hsv_to_hex(hues[0], sats[0], vals[0]))
    col3_disp.config(bg=hsv_to_hex(hues[1], sats[1], vals[1]))
    col4_disp.config(bg=hsv_to_hex(hues[2], sats[2], vals[2]))
    col5_disp.config(bg=hsv_to_hex(hues[3], sats[3], vals[3]))



# WIP
# get colors from new boxes & change label to display hex values
def find_hex():
    col1 = col1_disp.cget("bg")
    col1_code.config(text=col1)


# base frame & text
root = tk.Tk()
root.title("Generate Palette from Color")

inst = tk.Label(root, text="Select a color & harmony type\nto generate a palette")
inst.grid(column=2, row=0)

pal_label = tk.Label(root, text="Current Palette:")
pal_label.grid(column=2, row=3)

# dropdown generation
image_paths = {
    "i Type": "matsuda-i.jpg",  # Replace with actual paths
    "I Type": "matsuda-big-i.jpg",
    "V Type": "matsuda-V.jpg",
    "T Type": "matsuda-T.jpg",
    "X Type": "matsuda-X.jpg",
    "L Type": "matsuda-L.jpg",
    "Y Type": "matsuda-Y.jpg",
    "N Type": "matsuda-N.jpg"
}

options = list(image_paths.keys())

combo = ttk.Combobox(root, values=options)
combo.grid(column=3, row=1)
combo.bind("<<ComboboxSelected>>", change_image)

# Initial placeholder image
default_img = Image.new("RGB", (128, 128), "white")
default_photo = ImageTk.PhotoImage(default_img)

image_label = tk.Label(root, image=default_photo)
image_label.grid(column=3, row=0)
image_label.image = default_photo

# color display
frame = tk.Frame(root, width=128, height=128, bg="white")
frame.grid(column=1, row=0)

# buttons
change_color_button = tk.Button(root, text="Select Color", command=change_color)
change_color_button.grid(column=1, row=1)

generate_palette_button = tk.Button(root, text="Generate Palette", command=gen_palette)
generate_palette_button.grid(column=2, row=2)

# palette display & hex
col1_disp = tk.Frame(root, width=128, height=128, bg="white")
col1_disp.grid(column=0, row=4)
col2_disp = tk.Frame(root, width=128, height=128, bg="white")
col2_disp.grid(column=1, row=4)
col3_disp = tk.Frame(root, width=128, height=128, bg="white")
col3_disp.grid(column=2, row=4)
col4_disp = tk.Frame(root, width=128, height=128, bg="white")
col4_disp.grid(column=3, row=4)
col5_disp = tk.Frame(root, width=128, height=128, bg="white")
col5_disp.grid(column=4, row=4)

col1_code = tk.Label(root, text="")
col2_code = tk.Label(root, text="")
col3_code = tk.Label(root, text="")
col4_code = tk.Label(root, text="")
col5_code = tk.Label(root, text="")

column_size = 150
root.columnconfigure(0, minsize=column_size)
root.columnconfigure(1, minsize=column_size)
root.columnconfigure(2, minsize=column_size)
root.columnconfigure(3, minsize=column_size)
root.columnconfigure(4, minsize=column_size)
root.mainloop()
