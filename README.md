## Resolving inter-regional communication capacity in the human connectome

This repository contains code written in support of the work presented in 
[Resolving inter-regional communication capacity in the human connectome](https://www.biorxiv.org/content/10.1101/2022.09.28.509962v1), 
as well as data necessary to reproduce the results.

We developped a simple standardization method to investigate polysynaptic communication pathways between pairs of cortical regions.  
We then leveraged this procedure to relate inter-regional and regional patterns of communication propensity to 
the canonical intrinsic functional organization of the human cortex, meta-analytic probabilistic patterns of functional specialization, and nodal topological features.

Each of the main directories of this repository are documented with a README file, so feel free to check them out for more details.

### Running the analysis

1. Git clone this repository.
2. Download the necessary preprocessed data files from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7150367.svg)](https://doi.org/10.5281/zenodo.7150367) and place them in the appropriate [`preprocessed_data`](https://github.com/fmilisav/milisav_dyadic_communication/tree/main/data/preprocessed_data) folder. 
3. Install the relevant Python packages by building and activating a conda environment from the provided environment.yml file. To do so, in the command line, type:

```bash
cd milisav_dyadic_communication
conda env create -f environment.yml
conda activate milisav_dyadic_communication
```

4. To run the analysis, simply type:

```bash
python code/analysis/analysis.py
```
