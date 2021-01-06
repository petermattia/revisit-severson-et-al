# revisit-severson-et-al

**NOTE:** Please contact Prof. Richard Braatz, braatz@mit.edu, for access to the code repository associated with the [Severson et al. publication in *Nature Energy*](https://doi.org/10.1038/s41560-019-0356-8) (available with an academic license). This repository is not directly related to the *Nature Energy* paper.

This repository contains code for our work entitled "Statistical learning for accurate and interpretable battery lifetime prediction", a follow-up paper to [Severson et al.](https://doi.org/10.1038/s41560-019-0356-8) A permanent archive of this work on Zenodo is available here:
[![DOI](https://zenodo.org/badge/282795046.svg)](https://zenodo.org/badge/latestdoi/282795046)

Our key scripts and functions are summarized here:
- `featuregeneration.m`: MATLAB script that generates capacity arrays from the [battery dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and exports them to csvs (stored in `/data`).
- `revisit-severson-et-al.ipynb`: Python notebook containing most of the analysis and figure generation.
- `image_annotated_heatmap.py`: Helper function from matplotlib (see docstring for source).
