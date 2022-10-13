# `code`

- [`utils.py`](./code/utils.py) contains generic auxiliary functions used in preprocessing and analysis scripts.

## `analysis`

The [`analysis`](./code/analysis) folder contains the code used to conduct the analyses and generate the plots available in [`figures`](./figures).  
[`analysis.py`](./code/analysis/analysis.py) is the main analytic script. It makes use of auxiliary functions available in [`analysis_utils.py`](./code/analysis/analysis_utils.py). 

## `preprocessing`

The [`preprocessing`](./code/preprocessing) folder contains the code used to preprocess data from the Lausanne dataset, initially released [HERE](https://doi.org/10.5281/zenodo.2872624),
yielding the preprocessed data files available in [`preprocessed_data`](./data/preprocessed_data) and used to run the analysis.

- [`match_length_degree_distribution.py`](./code/preprocessing/match_length_degree_distribution.py) contains a function for generating degree- and edge length-preserving surrogate connectomes.
The original MATLAB code written by Rick Betzel is available [HERE](https://www.brainnetworkslab.com/coderesources). This version of the code was translated to Python by Justine Hansen.

- [`path_transitivity.py`](./code/preprocessing/path_transitivity.py) contains a function for computing path transitivity between all pairs of nodes in an undirected connectivity matrix.
It was translated to Python from the MATLAB function `path_transitivity.m`, openly available in the [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet) 
and originally written by Olaf Sporns, Andrea Avena-Koenigsberger and Joaquin Goñi.

- [`preprocessing.py`](./code/preprocessing/preprocessing.py) is the main preprocessing script.

- [`strength_preserving_rand.py`](./code/preprocessing/strength_preserving_rand.py) contains a function for the degree- and strength-preserving randomization of undirected, weighted adjacency matrices via simulated annealing.
It was adapted in Python from a MATLAB function written by Rick Betzel for [Misic et al., 2015](https://doi.org/10.1016/j.neuron.2015.05.035).
