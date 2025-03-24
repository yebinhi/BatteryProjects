import torch
from torch import nn

class LSTMSOHModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMSOHModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length=1, input_size)
        LSTM outputs shape: (batch_size, sequence_length=1, hidden_size)
        """
        out, _ = self.lstm(x)  # LSTM expects 3D input
        out = self.fc(out[:, -1, :])  # Take the last time-step's output
        return out
