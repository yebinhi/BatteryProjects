from torch import nn


class LinearSOHModel(nn.Module):
    def __init__(self, input_size):
        super(LinearSOHModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x.squeeze(1))