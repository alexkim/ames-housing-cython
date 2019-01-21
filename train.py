"""Machine Learning on the Ames Housing Dataset

Alex Kim
Python 3.7

This module runs the machine learning scripts written in Cython.
"""
import pyximport; pyximport.install()
import numpy as np
import ridge

def split_data():
    """Splits the full training dataset into a training set and a
    validation set (4:1 ratio).
    """

if __name__ == "__main__":
    data = np.genfromtxt("data/train_cleaned.csv", delimiter=",",
            skip_header=1)    
    data = np.delete(data, 0, 1)  # Delete ID column

    weights = ridge.train(data, num_epochs=200, step_size=0.00000000001,
            lam=0.01)
    ridge.test(data, weights)
