# Color Harmony Project

## Files:
color_network.py
- requires a connection_matrix.csv created by connection_mining.py
- creates a html file using pyvis.network that shows pairwise color connections.
- NUM_PALETTES: number of kimonos to analyze
- NODE_SCALE: the size of nodes
- EDGE_WEIGHT_SCALE: how much to increase the weight of the edge if the number of connections increases
- EDGE_FREQ_CUTOFF: removes edges below a this frequency
- EDGE_RAND_CUTOFF: randomly prunes this many edges

connection_mining.py
- requires kimono images in /Kimono_Images/image{image_number}.jpg
- creates connection_matrix.csv
- DIMENSION: dimension of bucketing. Usually DIMENSION=8 is used.
- connection_matrix.csv is a csv containing pairwise affinity scores for all colors across all kimonos. This score was calculated as such: given a kimono with x pixels of color A and y pixels of color B, a score of xy is added to the cumulative A x B color affinity score.
- Colors are chosen by bucketing the entirety of RGB space into DIMENSION^3 number of buckets. All colors within a bucket are counted as occurences of that bucket's color.



