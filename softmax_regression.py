import math
import os
import random

import numpy as np
import torch
from torch import nn
from torch import optim

from utils.plot_utils import plot_model

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


class SoftmaxRegressionNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2, 10)
        self.activation1 = nn.ReLU()
        self.hidden1 = nn.Linear(10, 10)
        self.activation2 = nn.ReLU()
        self.hidden2 = nn.Linear(10, 10)
        self.activation3 = nn.ReLU()
        self.output = nn.Linear(10, 3)

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


def generate_spiral_data(N, D, C, device):
    # (3000, 2)
    # 1000 samples per class
    X = torch.zeros(N * C, D).to(device)
    y = torch.zeros(N * C, dtype=torch.long).to(device)

    for c in range(C):
        index = 0
        t = torch.linspace(0, 1, N)
        # When c = 0 and t = 0: start of linspace
        # When c = 0 and t = 1: end of linpace
        # This inner_var is for the formula inside sin() and cos() like sin(inner_var) and cos(inner_Var)
        inner_var = torch.linspace(
            # When t = 0
            (2 * math.pi / C) * (c),
            # When t = 1
            (2 * math.pi / C) * (2 + c),
            N
        ) + torch.randn(N) * 0.2

        for ix in range(N * c, N * (c + 1)):
            X[ix] = t[index] * torch.FloatTensor((
                math.sin(inner_var[index]), math.cos(inner_var[index])
            ))
            y[ix] = c
            index += 1

    return X, y


def train_model_for_spiral_data(can_plot=False):
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    N = 1000  # num_samples_per_class
    D = 2  # dimensions
    C = 3  # num_classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Both data and model should be on the device
    X, y = generate_spiral_data(N, D, C, device)
    model = SoftmaxRegressionNN().to(device)

    lr = 1e-3
    epochs = 1000

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create training loop
    for epoch in range(epochs):
        y_hat = model(X)
        loss = criterion(y_hat, y)

        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        p_hat_classes = torch.max(y_hat.detach(), axis = 1)  # Detach y_hat from the computation graph
        accuracy = (p_hat_classes.indices == y).sum().item() / y.size(0)
        print(f'Accuracy: {accuracy:.4f}')

        # Zero out grad on each training epoch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if can_plot:
        plot_model(X, y, model)


if __name__ == '__main__':
    model = SoftmaxRegressionNN()
    print(model)

    for name, param in model.named_parameters():
        print(name, param.shape)

    train_model_for_spiral_data(can_plot=True)
