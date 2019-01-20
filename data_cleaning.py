"""Machine Learning on the Ames Housing Dataset

Alex Kim
Python 3.7

This module cleans the training dataset.
"""
import pandas as pd
import argparse

def clean(csv_path):
    # Load DataFrame from CSV path
    data = pd.read_csv(csv_path)

   # Convert categorical variables to binary "dummy" variables
    data = pd.get_dummies(data, dummy_na=True)

    # Move response variable to end
    new_col_order = list(data.columns)
    new_col_order.remove("SalePrice")
    new_col_order.append("SalePrice")
    data = data[new_col_order]

    # Export data to a new CSV
    data.to_csv("data/train_cleaned.csv", index=False)

if __name__ == "__main__":
    clean("data/train.csv")
