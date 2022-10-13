# `data`

- [`cammoun_ns`](./data/original_data/cammoun_ns) contains `.csv` files corresponding to 
[Neurosynth](https://github.com/neurosynth/neurosynth) functional activation maps 
for 123 terms from the [Cognitive Atlas](https://www.cognitiveatlas.org/) 
parcellated according to both the 219 (scale125) and 1000 (scale500) nodes resolutions of the Cammoun atlas.

## `preprocessed_data`

The [`preprocessed_data`](./data/preprocessed_data) folder contains preprocessed `.pickle` data files used to perform the analyses.

### Consensus networks

- [`struct_den_dict`](./data/preprocessed_data/struct_den_dict.pickle) contains the structural consensus networks of the original (1000 nodes - *Discovery* dataset - weighted, labelled `500_discov_wei`) 
and the sensitivity datasets (1000 nodes - *Validation* dataset - weighted, labelled `500_valid_wei`; 219 nodes - *Discovery* dataset - weighted, labelled `125_discov_wei`; 1000 nodes - *Discovery* dataset - binary, labelled `500_discov_bin`).

- [`func_dict`](./data/preprocessed_data/func_dict.pickle) contains the functional consensus network of the original dataset.

- [`rand_struct_den`](./data/preprocessed_data/rand_struct_den.pickle) contains three examples of null structural consensus networks derived from the original dataset using a strength sequence-preserving approach.

### Communication matrices

- [`dist_dicts`](https://doi.org/10.5281/zenodo.7150367) contains empirical communication distance matrices across all communication models considered in this project 
computed on the original and the sensitivity structural consensus networks.

- [`rand_dist`](./data/preprocessed_data/rand_dist.pickle) contains three examples of null shortest path length matrices derived from the original dataset 
using a strength sequence-preserving approach.

- [`mean_dist_dicts`](https://doi.org/10.5281/zenodo.7150367) contains mean null communication distance matrices across all communication models considered in this project 
computed on the original and the sensitivity null structural consensus networks randomized according to a strength sequence-preserving approach.

- [`std_dist_dicts`](https://doi.org/10.5281/zenodo.7150367) contains matrices of standard deviations across null communication distance measures from all communication models considered in this project 
computed on the original and the sensitivity null structural consensus networks randomized according to a strength sequence-preserving approach.

- [`geo_mean_dist_dicts`](./data/preprocessed_data/geo_mean_dist_dicts.pickle) contains mean null communication distance matrices across all communication models considered in this project 
computed on the original and the sensitivity null structural consensus networks randomized according to a geometry-preserving approach.

- [`geo_std_dist_dicts`](./data/preprocessed_data/geo_std_dist_dicts.pickle) contains matrices of standard deviations across null communication distance measures from all communication models considered in this project 
computed on the original and the sensitivity null structural consensus networks randomized according to a geometry-preserving approach.

- [`norm_dist_dicts`](https://doi.org/10.5281/zenodo.7150367) contains standardized communication distance matrices across all communication models considered in this project 
for the original and the sensitivity datasets. The standardization procedure was based on strength sequence-preserving nulls.

- [`geo_norm_dist_dicts`](./data/preprocessed_data/geo_norm_dist_dicts.pickle) contains standardized communication distance matrices across all communication models considered in this project 
for the original and the sensitivity datasets. The standardization procedure was based on geometry-preserving nulls.

### Communication arrays

- [`node_mean_norm_dist_dicts`](./data/preprocessed_data/node_mean_norm_dist_dicts.pickle) contains node-wise mean standardized communication distance measures across all communication models considered in this project 
for the original and the sensitivity datasets. The standardization procedure was based on strength sequence-preserving nulls.

- [`node_mean_geo_norm_dist_dicts`](./data/preprocessed_data/node_mean_geo_norm_dist_dicts.pickle) contains node-wise mean standardized communication distance measures across all communication models considered in this project 
for the original and the sensitivity datasets. The standardization procedure was based on geometry-preserving nulls.

### Topological and geometric features

- [`strengths_dict`](./data/preprocessed_data/strengths_dict.pickle) contains node-wise strength measures 
computed on the original and the sensitivity structural consensus networks.
 
- [`betweenness_dict`](./data/preprocessed_data/betweenness_dict.pickle) contains node-wise betweenness centrality measures 
computed on the original and the sensitivity structural consensus networks.

- [`participation_dict`](./data/preprocessed_data/participation_dict.pickle) contains node-wise participation coefficients 
computed on the original and the sensitivity structural consensus networks.

- [`euc_dist125`](./data/preprocessed_data/euc_dist125.pickle) and [`euc_dist500`](./data/preprocessed_data/euc_dist500.pickle) 
contain matrices of pairwise Euclidean distance among nodes of the 219 and 1000 nodes Cammoun parcellations, respectively.

### MISC

- [`mask125`](./data/preprocessed_data/mask125.pickle) and [`mask500`](./data/preprocessed_data/mask500.pickle) contain masks for extracting 
the upper triangles of 219x219 and 1000x1000 matrices, respectively, in line with the two Cammoun resolutions considered.

- [`cammoun_ns_df_idx`](./data/preprocessed_data/cammoun_ns_df_idx.pickle) contains indices for mapping Cammoun parcels from 
the [Neurosynth maps](./data/original_data/cammoun_ns) to the [netneurotools info](https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.datasets.fetch_cammoun2012.html#netneurotools.datasets.fetch_cammoun2012)
for both the 219 (scale125) and 1000 (scale500) nodes resolutions of the Cammoun atlas.




