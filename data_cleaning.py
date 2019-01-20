"""Machine Learning on the Ames Housing Dataset

Alex Kim
Python 3.7

This module cleans the training dataset.
"""
import pandas as pd
import argparse

def clean(csv_path):
    # Load data from CSV path
    data = pd.read_csv(csv_path)

    # Convert categorical variables to binary "dummy" variables
    data = pd.get_dummies(data, dummy_na=True)

    # Export data to a new CSV
    data.to_csv("data/train_cleaned.csv")

if __name__ == "__main__":
    clean("data/train.csv")
