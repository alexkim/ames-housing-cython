"""Machine Learning on the Ames Housing Dataset

Alex Kim
Python 3.7

This module defines a ridge regression algorithm that uses stochastic
gradient descent.
"""
import numpy as np

def get_loss_gradient():
    """Computes the numerical gradient of the ridge loss function with
    respect to the feature weight vector.

    Args:
        features (dict)

    Returns:
        A NumPy vector representing the gradient of the loss function.
    """
    loss_gradient = np.zeros()
    pass

def train_model(train_data, num_iters, step_size, lam):
    """Trains linear regression model.

    Returns:
        A NumPy vector representing the regression weights.
    """
    pass

def test_model(test_data):
    pass
