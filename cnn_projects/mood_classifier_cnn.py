import torch
from PIL import Image
from torch import nn

from cnn_utils import load_happy_dataset
from cnn_utils import predict, create_device, training_loop, test_single

"""
## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
## Conv2D with 32 7x7 filters and stride of 1
## BatchNormalization for axis 3
## ReLU
## Max Pooling 2D with default parameters
## Flatten layer
## Dense layer with 1 unit for output & 'sigmoid' activation
"""


class MoodClassifierCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),  # => 64x64x32
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # => 32x32x32
            nn.Flatten(),
            nn.Linear(in_features=32 * 32 * 32, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == '__main__':
    cnn = MoodClassifierCNN()

    # Load data
    train_X, train_y, test_X, test_y, classes = load_happy_dataset()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    training_loop(cnn, criterion, optimizer, train_X, train_y)
    test_single(cnn, test_X, test_y)

    # Test a sad image
    sad_mask = (test_y.view(-1) == 0)
    sad_X = torch.unsqueeze(test_X[sad_mask][0], dim=0)
    sad_y = test_y[sad_mask][0]
    sad_y_pred = predict(cnn, sad_X.to(create_device()))

    print(f"y = {sad_y.item()}, y_pred = {sad_y_pred.item()}")
    img = Image.fromarray(torch.squeeze(sad_X, dim=0).cpu().numpy().astype('uint8').transpose([1, 2, 0]))
    img.show()
