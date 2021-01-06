# revisit-severson-et-al

**NOTE:** Please contact Prof. Richard Braatz, braatz@mit.edu, for access to the code repository associated with the Nature Energy paper (available with an academic license). This repository is unrelated to the Nature Energy paper.

This repository contains code for our work entitled "Statistical learning for accurate and interpretable battery lifetime prediction".

Our key scripts and functions are summarized here:
- `featuregeneration.m`: MATLAB script that generates capacity arrays from the [battery dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and exports them to csvs (stored in `/data`).
- `revisit-severson-et-al.ipynb`: Python notebook containing most of the analysis and figure generation.
- `image_annotated_heatmap.py`: Helper function from matplotlib (see docstring for source).
