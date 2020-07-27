# predicting battery lifetime

**NOTE:** Please contact Prof. Richard Braatz, braatz@mit.edu, for access to the code repository associated with the Nature Energy paper (available with an academic license). This respository is unrelated to the Nature Energy paper.

This repository contains code for our work on early prediction of battery lifetime. Features are generated in MATLAB, while the machine learning is performed in python using numpy, scikit-learn, and matplotlib.

Our key scripts and functions are summarized here:

MATLAB code:
- featuregeneration.m: Generates large set of features extracted from battery dataset and exports them to csvs. This function loops through cycles 20 through 100 in increments of 10.

Python code:
- ElasticNet.py: uses the ElasticNetCV module from scikit-learn to train an elastic net model.  The module automatically performs 5-fold cross-validation to select the optimal values for the alpha and l-1 ratio hyperparameters.  The hyperparameters are individually optimized for each number of initial cycles used (20 through 100 in steps of 10). Saves the trained models so they can be used for testing.
- RandomForest.py: uses the RandomForestRegressor module from scikit-learn.  For each number of cycles, we do a grid-search over the n_trees and max_depth hyperparameters, and use 5-fold cross-validation to calculate the mean percent error and mean standard error.  We select the best hyperparameters separately for each number of cycles. Saves the trained models so they can be used for testing.
- Adaboost.py: uses the AdaBoostRegressor module from scikit-learn.  We performed a grid-search over the n_trees and learning_rate hyperparameters to select the optimal values.  For each number of cycles we use 5-fold cross-validation to calculate the mean percent error and mean standard error. Saves the trained models so they can be used for testing.
- SVR.py: uses the SVR module from scikit-learn to perform support vector regression.  We did some preliminary exploration using support vector regression with a few different kernels, but found that it was extremely prone to overfitting.  We decided to switch to other regression methods instead.
- coeff_plotter.py: creates a colormap plot showing the relative weights/importance of the different features for the trained models. 
- test.py: Runs trained models on test data and generates plots of the results
