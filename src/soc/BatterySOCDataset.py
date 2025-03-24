import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.Config import Config


class BatterySOCDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = pd.read_csv(X_path).values.reshape(-1, 1, Config.INPUT_SIZE)
        self.y = pd.read_csv(y_path).values.reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)