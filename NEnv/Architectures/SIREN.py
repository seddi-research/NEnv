import numpy as np
import torch
import torch.nn as nn

"""
SIREN implementation taken from https://github.com/vsitzmann/siren 
"""


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        """

        @param in_features: Number of input variables
        @type in_features: int
        @param out_features: Number of output variables
        @type out_features: int
        @param bias: Set to True if you want to learn a bias parameter in this layer
        @type bias: bool
        @param is_first: Set to True if this is the first layer of the siren
        @type is_first: bool
        @param omega_0: Parameter that weights the initialization of the layer
        @type omega_0: float
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        """
        @param in_features: Number of input variables to the model
        @type in_features: int
        @param hidden_features: Number of hidden neurons in each layer
        @type hidden_features: int
        @param hidden_layers: Number of hidden layers
        @type hidden_layers: int
        @param out_features: Number of output variables of the model
        @type out_features: int
        @param outermost_linear: Set to true if you want a linear layer at the end
        @type outermost_linear: bool
        @param first_omega_0: Parameter that weights the initialization of the first layer
        @type first_omega_0: float
        @param hidden_omega_0: Parameter that weights the initialization of the other layers
        @type hidden_omega_0: float
        """

        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
