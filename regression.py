import os
import random

import numpy as np
import torch
from torch import nn
from torch import optim

from plot_utils import scatter_plot_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
Create a neural network for multinomial logistic regression.
The data is in R^2 and has 3 classes. The data is not linearly separable.

Neural network architecture:
input layer: 2 units (Since the data is in R^2)
hidden layer 1: 10 units
hidden layer 2: 10 units
output layer: 3 units (Since there are 3 classes)

The output layer has softmax activation function.
'''


class RegressionNN(nn.Module):

    def __init__(self, use_relu=True):
        super().__init__()

        self.input = nn.Linear(1, 10)
        self.activation1 = nn.ReLU() if use_relu else nn.Tanh()
        self.hidden1 = nn.Linear(10, 10)
        self.activation2 = nn.ReLU() if use_relu else nn.Tanh()
        self.hidden2 = nn.Linear(10, 10)
        self.activation3 = nn.ReLU() if use_relu else nn.Tanh()
        self.output = nn.Linear(10, 1)

        self.network = nn.Sequential(
            self.input,
            self.activation1,
            self.hidden1,
            self.activation2,
            self.hidden2,
            self.activation3,
            self.output
        )

    def forward(self, x):
        x = self.network(x)
        return x


def generate_data(N, device):
    X = torch.unsqueeze(torch.linspace(-1, 1, N), dim=1).to(device)
    y = X.pow(3) + 0.3 * torch.rand(X.size()).to(device)

    return X, y


def train_model(model, can_plot=False):
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    N = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Both data and model should be on the device
    X, y = generate_data(N, device)
    model = model.to(device)

    lr = 1e-3
    epochs = 1000

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create training loop
    for epoch in range(epochs):
        y_hat = model(X)
        loss = criterion(y_hat, y)

        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # Zero out grad on each training epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if can_plot:
        scatter_plot_model(X, y, model)


if __name__ == '__main__':
    #train_model(RegressionNN(use_relu=True), can_plot=True)     # Uses piecewise functions to approx the data
    train_model(RegressionNN(use_relu=False), can_plot=True)    # Uses smooth fn to approx the data
