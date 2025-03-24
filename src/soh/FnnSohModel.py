import torch
from torch import nn


class FNNSOHModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3):
        super(FNNSOHModel, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch_size, input_size)
        """
        return self.model(x)


# # Example usage
# if __name__ == "__main__":
#     batch_size = 32
#     input_size = 10  # Example feature size
#     model = FNNSOHModel(input_size)
#
#     sample_input = torch.randn(batch_size, input_size)  # Random input tensor
#     output = model(sample_input)
#     print(output.shape)  # Should be (batch_size, 1)
