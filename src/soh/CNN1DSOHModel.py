from torch import nn


class CNN1DSOHModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DSOHModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, 1)

    # def forward(self, x):
    #     x = x.transpose(1, 2)  # to (batch, channels=1, features)
    #     x = self.conv(x)
    #     x = x.view(x.size(0), -1)
    #     return self.fc(x)

    def forward(self, x):
        x = x.squeeze(1)  # (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)