"""Machine Learning on the Ames Housing Dataset

Alex Kim
Python 3.7

This module defines a ridge regression algorithm that uses stochastic
gradient descent.
"""
import numpy as np

def get_loss_gradient(example, weights, lam):
    """Computes the numerical gradient of the ridge loss function with
    respect to the feature weight vector.

    Args:
        example (np.ndarray): A single example (1-dimensional), where
            the last element is the response variable
        lam (double): A tuning parameter for the ridge regularization
            term

    Returns:
        A NumPy vector representing the gradient of the loss function.
    """
    feature_vector = np.delete(example, -1, 0)
    response_value = example[-1]

    rss_gradient = 2 * (np.dot(feature_vector, weights) - response_value) \
                   * feature_vector
    l2_gradient = lam * weights
    loss_gradient = rss_gradient + l2_gradient

    return loss_gradient


def train(train_data, num_epochs, step_size, lam):
    """Trains the weights of the linear regression model.

    Args:
        train_data (np.ndarray): 
        num_iters (int):
        step_size (double):
        lam (double):

    Returns:
        A NumPy vector representing the regression weights.
    """
    num_examples = train_data.shape[0]
    num_features = train_data.shape[1] - 1
    weights = np.zeros(num_features)

    for epoch in range(num_epochs):
        sq_errors = []
        for i in range(num_examples):
            example = train_data[i]
            loss_gradient = get_loss_gradient(example, weights, lam)
            weights = weights - (step_size * loss_gradient)

            feature_vector = np.delete(example, -1, 0)
            response_value = example[-1]
            sq_errors.append((np.dot(weights, feature_vector) - response_value) ** 2)
        mean_sq_error = sum(sq_errors) / num_examples
        print(mean_sq_error)

    return weights


def test(test_data, weights):
    """Tests the regression weights on a test dataset.

    Returns:
        The mean squared error (double) of the predictions on the given
        test dataset.
    """
    inputs = np.delete(test_data, -1, 1)
    outputs = test_data[:, -1]

    predictions = np.dot(inputs, weights)
    mean_sq_error = np.sum((predictions - outputs) ** 2) / inputs.shape[0]
    print(mean_sq_error)
