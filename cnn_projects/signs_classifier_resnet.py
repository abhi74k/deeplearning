import torch
from torch import nn

from cnn_projects import cnn_utils
from cnn_utils import load_signs_dataset
from cnn_utils import training_loop, test_single
from Resnet import ResNet

if __name__ == '__main__':

    # Load data
    train_X, train_y, test_X, test_y, classes = load_signs_dataset()

    cnn = ResNet(classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    training_loop(cnn, criterion, optimizer, train_X, train_y, accuracy_fn=cnn_utils.calculate_multi_class_accuracy, epochs=30)

    for i in range(5):
        test_single(cnn, test_X, test_y, classification='multi')
