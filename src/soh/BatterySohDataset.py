import torch
from torch.utils.data import Dataset
from BatteryDataPreparer import BatteryDataPreparer

class BatteryDataset(Dataset):
    def __init__(self, file_paths, min_percent=0.2, max_percent=0.8):
        preparer = BatteryDataPreparer(file_paths, min_percent, max_percent)
        self.X, self.y = preparer.prepare_data()

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

        # Reshape X to (batch_size, sequence_length=1, input_size)
        self.X = self.X.unsqueeze(1)  # Adds a "sequence length" of 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
