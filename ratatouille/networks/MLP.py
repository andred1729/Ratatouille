import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU):
        """
        Initializes a Multi-Layer Perceptron (MLP) network with Xavier initialization.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            hidden_dims (list of int): List of hidden layer dimensions.
            activation (nn.Module): Activation function to use (default: nn.ReLU).
        """
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(current_dim, hidden_dim)
            init.xavier_uniform_(linear_layer.weight)  # Xavier initialization
            layers.append(linear_layer)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim

        final_layer = nn.Linear(current_dim, output_dim)
        init.xavier_uniform_(final_layer.weight)  # Xavier initialization
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)