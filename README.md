# revisit-severson-et-al

**NOTE:** Please contact Prof. Richard Braatz, braatz@mit.edu, for access to the code repository associated with the [Severson et al. publication in *Nature Energy*](https://doi.org/10.1038/s41560-019-0356-8) (available with an academic license). This repository is not directly related to the *Nature Energy* paper.

This repository contains code for our work entitled "[Statistical learning for accurate and interpretable battery lifetime prediction](https://arxiv.org/abs/2101.01885)", a follow-up paper to [Severson et al.](https://doi.org/10.1038/s41560-019-0356-8) A permanent archive of this work on Zenodo is available here:
[![DOI](https://zenodo.org/badge/282795046.svg)](https://zenodo.org/badge/latestdoi/282795046)

Our key scripts and functions are summarized here:
- `featuregeneration.m`: MATLAB script that generates capacity arrays from the [battery dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and exports them to csvs (stored in `/data`).
- `revisit-severson-et-al.ipynb`: Python notebook containing most of the analysis and figure generation.
- `image_annotated_heatmap.py`: Helper function from matplotlib (see docstring for source).
- `lifetime_charging_time/lifetime_charging_time.ipynb`: Contains a mini analysis on cycle life vs. charging time.

In addition, we used three scripts for neural net model training in the `nn_models` directory.
Calls to these scripts look like this:
`python mlp_base.py --n_starts 10 --lr 0.001 --rw 0.0001 --hd 10`.
- `mlp_base.py`: Trains MLP models
- `cnn_base.py`: Trains CNN models without baseline subtraction
- `cnn_base_nosubtract.py`: Trains CNN models with baseline subtraction

These scripts are used by another script that is written
in a proprietary language used to interact with a cluster at IBM;
this additional script is not included.

Finally, we include three notebooks that analyze the results of MLP & CNN training
in the `nn_models` directory.
These notebooks will not run completely without all `.pkl` or `.pt` files present.
- `Analyze_Results_MLP_CV.ipynb`: Analyzes the results from the MLP cross-validation study and selects the best hyperparameters. Also produces intermediate outputs including `mlp_predictions.json`, `mlp_predictions_cycavg.json`, and `mlp_shap_results.csv`.
- `MLP_FinalModel.ipynb`: Runs the final model based on the results of CV and generates the MLP interpretability figure using [`shap`](https://github.com/slundberg/shap).
-`Analyze_Results_CNN.ipynb`: Analyzes the results from the CNN study
(not a formal CV study) and selects the best hyperparameters. The best performing model (`CNN_n20000_rw0.001_lr0.0001_do0.0.pt`) 
is included for reference.
