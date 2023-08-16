import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_model(X, y, model):
    model.cpu()
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():  # Don't calculate gradients when the model is evaluated
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()  # Z = model(data).cpu().numpy() Totally valid

    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(X, y)


def scatter_plot_model(X, y, model):
    model.cpu()
    with torch.no_grad():  # Don't calculate gradients when the model is evaluated
        z = model(X).detach()  # Z = model(data).cpu().numpy() Totally valid

    plt.plot(X, y, 'o', color='red', label='true')
    plt.plot(X, z, color='green', linestyle='--', label='predicted')
    plt.show()


def plot_data(X, y):
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
