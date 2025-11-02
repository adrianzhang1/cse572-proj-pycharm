import random

import pandas as pd
from pyvis.network import Network
import numpy as np
import math
import random

import itertools

#net parameters
NUM_PALETTES = 4522 #4522 max
NODE_SCALE = 20
EDGE_WEIGHT_SCALE = 0.01
EDGE_FREQ_CUTOFF = 10_000_000_000
EDGE_RAND_CUTOFF = 0

# index is an octal value of RGB constrained to 512 options
node_sizes = [1 for _ in range(512)]
edge_sizes = [[1 for _ in range(512)] for _ in range(512)]

colors_as_hex = [""]*512
for r in range(8):
    for g in range(8):
        for b in range(8):
            color = (r*32+16)*65536 + (g*32+16)*256 + (b*32+16)
            colors_as_hex[r*64 + g*8 + b] =f"#{color:06X}"
# proportions =


net = Network(height='1300px', width='80%')
# net.show_buttons(filter_=['physics'])
# net.barnes_hut()

node_sizes = [math.sqrt(size) * NODE_SCALE for size in node_sizes]
net.add_nodes(list(range(512)),title=colors_as_hex,color=colors_as_hex,size=node_sizes)

# net.add_edge(pair[0], pair[1], width=EDGE_WEIGHT_SCALE*(edge_sizes[pair[0]][pair[1]]+edge_sizes[pair[1]][pair[0]])) #
# num_edges += 1

num_edges = 0

connection_matrix = np.loadtxt("connection_matrix.csv", delimiter=",", dtype=int)
num_connections_per_color = [0 for _ in range(512)]
for i in range(1,511):
    for j in range(1,511):
        num_connections_per_color[i] += connection_matrix[i][j]

EDGE_PROP_CUTOFF = 0.05

for i in range(1,511):
    for j in range(i+1,511):
        # if connection_matrix[i][j]/num_connections_per_color[i] > EDGE_PROP_CUTOFF:
        if connection_matrix[i][j] > EDGE_FREQ_CUTOFF:
            net.add_edge(i,j)
            num_edges += 1

# nx_graph.remove_node(2)
# for i in range(512):
#     for j in range(i):
#         if edge_sizes[i][j] >= 1:
#             num_edges +=1

print("number of edges:", num_edges)



# for size in node_sizes:
#     if size>0:
#         print(size)
net.repulsion()
net.show("network_" +
         f"{EDGE_FREQ_CUTOFF:.1e}" + "freq" +
         ".html",notebook=False)


