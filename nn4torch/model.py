import torch.nn as nn


class NeuralNetwork(nn.Module):
    """A simple multi-layer perceptron with two hidden layers."""

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        """
        Initializes the network layers.
        """
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            # Hidden Layer 1
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Hidden Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Output Layer
            nn.Linear(hidden_size, output_size),
        )
        # Initialize weights using He initialization, which is good for ReLU
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        """
        Performs the forward pass.
        Softmax is not applied here because nn.CrossEntropyLoss applies it internally.
        """
        return self.network(x)
