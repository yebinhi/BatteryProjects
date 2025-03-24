import pandas as pd
import numpy as np

class BatteryDataPreparer:
    def __init__(self, file_path, min_percent=0.2, max_percent=0.8, num_columns=101):
        """
        Initializes the data preparer with a single file path and feature selection range.

        :param file_path: Path to the combined dataset (all.csv).
        :param min_percent: Starting percentage for feature selection.
        :param max_percent: Ending percentage for feature selection.
        :param num_columns: Expected number of columns in the dataset.
        """
        self.file_path = file_path
        self.min_percent = min_percent
        self.max_percent = max_percent
        self.num_columns = num_columns  # Ensuring the correct number of columns

    def load_data(self):
        """
        Loads the combined dataset from a single CSV file (no header).

        :return: Pandas DataFrame containing the dataset.
        """
        # Read CSV without headers and set column names manually
        data = pd.read_csv(self.file_path, header=None)

        # Ensure the dataset has exactly 101 columns
        if data.shape[1] != self.num_columns:
            raise ValueError(f"Expected dataset with {self.num_columns} columns, but found {data.shape[1]}.")

        return data

    def prepare_data(self):
        """
        Prepares training and target data by selecting features based on min/max percentages.

        :return: Tuple (X, y) where:
                 - X is the training feature matrix (features only).
                 - y is the target SOH column.
        """
        raw_data = self.load_data()
        if raw_data.empty:
            raise ValueError("Loaded dataset is empty. Check file path and contents.")

        num_features = self.num_columns - 1  # Exclude last column (SOH)
        start_index = int(num_features * self.min_percent)
        end_index = int(num_features * self.max_percent)

        # Ensure valid feature selection range
        if start_index >= end_index:
            raise ValueError(f"Invalid feature selection range: ({start_index}, {end_index})")

        X = raw_data.iloc[:, start_index:end_index]  # Select range of features
        y = raw_data.iloc[:, -1]  # Last column is SOH

        return X.values, y.values
