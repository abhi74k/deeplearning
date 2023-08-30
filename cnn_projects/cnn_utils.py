import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt


def load_h5_dataset(train_filename, test_filename):
    train_dataset = h5py.File(train_filename, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_x_orig = np.transpose(train_set_x_orig,
                                    [0, 3, 1, 2])  # Torch expects (batch_size, channels, width, height)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(test_filename, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_x_orig = np.transpose(test_set_x_orig, [0, 3, 1, 2])  # Torch expects (batch_size, channels, width, height)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = \
        torch.Tensor(train_set_x_orig), \
            torch.Tensor(train_set_y_orig), \
            torch.Tensor(test_set_x_orig), \
            torch.Tensor(test_set_y_orig), \
            torch.Tensor(classes)

    train_set_y_orig = torch.permute(train_set_y_orig, [1, 0])
    test_set_y_orig = torch.permute(test_set_y_orig, [1, 0])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_happy_dataset():
    return load_h5_dataset('datasets/train_happy.h5', 'datasets/test_happy.h5')


def load_signs_dataset():
    return load_h5_dataset('datasets/train_signs.h5', 'datasets/test_signs.h5')


def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = torch.round(y_pred)
        return y_pred


def calculate_binary_class_accuracy(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in data_loader:
            y_pred = model(X)
            y_pred = torch.round(y_pred)
            correct += (y_pred == y).sum().item()
            total += y.shape[0]
        accuracy = correct / total
        return accuracy


def calculate_multi_class_accuracy(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in data_loader:
            y_pred = model(X)
            y_pred_classes = torch.argmax(y_pred, dim=1)
            correct += (y_pred_classes.view(-1) == y.view(-1)).sum().item()
            total += y.shape[0]
        accuracy = correct / total
        return accuracy


def create_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def image_show(input_tensor, label=None, pause_duration=0.001):
    input_np = input_tensor.numpy().transpose([1, 2, 0])  # HWC
    mean = np.array([0.485, 0.456, 0.456])
    std = np.array([0.229, 0.224, 0.225])
    input_np = std * input_np + mean
    input_np = np.clip(input_np, 0, 1)
    plt.imshow(input_np)
    if label is not None:
        plt.title(label)
    plt.pause(pause_duration)


def training_loop(model, criterion, optimizer, train_X, train_y,
                  device=create_device(), epochs=50, batch_size=64,
                  accuracy_fn=calculate_binary_class_accuracy):
    train_dataset = TensorDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    training_loop_with_dataloader(model, criterion, optimizer, train_dataloader, device, epochs, accuracy_fn)


def training_loop_with_dataloader(model, criterion, optimizer, train_dataloader,
                                  device=create_device(), epochs=50,
                                  accuracy_fn=calculate_binary_class_accuracy,
                                  scheduler=None):
    model.to(device)

    for epoch in range(epochs):

        model.train()

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            """
            For binary classification(BCELoss), y andy_pred can both be floats
            
            For multi class classification, y_pred is an array of probabilities for each class i.e (batch size, num_classes).
            If y is of dimension (batchSize, 1) it should converted (batchSize,) and type uint8
            """
            if y_pred.shape[1] > 1 and y.shape[0] == y_pred.shape[0] and len(y.shape) > 1 and y.shape[1] == 1:
                y = y.view(-1).to(torch.uint8)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy_fn(model, train_dataloader)}')


def test_single(model, test_X, test_y, device=create_device(), classification='binary'):
    test_dataset = TensorDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_single_with_dataloader(model, test_dataloader, device, classification)


def test_single_with_dataloader(model, test_dataloader, device=create_device(), classification='binary', labels={}):
    model.to(device)

    X, y = next(iter(test_dataloader))
    X = X.to(device)
    y_test_pred = predict(model, X)
    y_test_pred.to('cpu')

    # Plot the image
    if classification == 'binary':
        print(f"y = {y.item()}, y_pred = {y_test_pred.item()}")
    else:
        print(f"y = {y.item()}, y_pred = {torch.argmax(y_test_pred).item()}")

    label = None
    if len(labels) > 0:
        label = labels[y.item()]

    image_show(torch.squeeze(X, dim=0).cpu(), label)