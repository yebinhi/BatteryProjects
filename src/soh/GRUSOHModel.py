from torch import nn


class GRUSOHModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUSOHModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out