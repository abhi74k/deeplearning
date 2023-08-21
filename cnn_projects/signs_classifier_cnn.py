import torch
from torch import nn

from cnn_projects import cnn_utils
from cnn_utils import load_signs_dataset
from cnn_utils import training_loop, test_single

"""
Implement the convolutional_model function below to build the following model: 
CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE. Use the functions above!

CONV2D: 8 filters 3x3, stride=1, padding=1
RELU
MAXPOOL: stride = 2, filter size = 2
CONV2D: 16 filters 3x3, stride=1, padding=1
RELU
MAXPOOL: window 4x4, stride 4, padding 'SAME'
FLATTEN
Dense layer: 6 neurons in output layer with "activation='softmax'" 
"""


class HandSignsClassifierCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # 64x64x3 => 64x64x8
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),  # => 8x8x8
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # 8x8x8=> 8x8x16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # => 2x2x16
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=6)
        )

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == '__main__':
    cnn = HandSignsClassifierCNN()
    print(cnn)

    for name, param in cnn.named_parameters():
        print(name, param.shape)

    # Load data
    train_X, train_y, test_X, test_y, classes = load_signs_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    training_loop(cnn, criterion, optimizer, train_X, train_y, accuracy_fn=cnn_utils.calculate_multi_class_accuracy, epochs=100)

    for i in range(5):
        test_single(cnn, test_X, test_y, classification='multi')
