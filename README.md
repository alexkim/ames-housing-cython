# Ames Housing Price Prediction

A Cython implementation of cross-validated ridge regression on the Ames Housing dataset. Based on a Python implementation in a [previous project](https://github.com/CerJesus/CS221FinalProject).

## TODO

 * Pass data/examples as tuples, not single array
 * Experiment with sparse data structures
 * Implement cross-validation on a given set of hyperparameters
 * Implement gradient boosting
 * Save MSE progress to a CSV
 * Incorporate optimization theory for step size


## Setup

 1. `mkdir data/`
 2. Save `train.csv` to `data/`
 3. `python data_cleaning.csv`
