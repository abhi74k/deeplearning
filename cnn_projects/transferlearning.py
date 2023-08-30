import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models

from cnn_projects import cnn_utils
from cnn_utils import training_loop_with_dataloader, test_single_with_dataloader, image_show

# In this project, we will use a pre-trained ResNet-18 model and fine-tune it to classify ants and bees.
# 2 modes of transfer learning is implemented
# 1. Finetuning the convnet: Instead of random initializaion, we initialize the network with a pretrained network,
# 2. ConvNet as fixed feature extractor: Freeze the weights for all of the network except that of the final fully

batch_size = 4

data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])
    ])
}

# Read and store train and validation set seperately
data = {
    x: datasets.ImageFolder('datasets/ants_bees_data/' + x, transform=data_transforms[x])
    for x in ['train', 'val']
}

# Define the data loaders for the training and validation sets
data_loader = {
    x: DataLoader(data[x], batch_size=batch_size if x == 'train' else 1, shuffle=True)
    for x in ['train', 'val']
}

train_data = data_loader['train']
val_data = data_loader['val']

batch_X, batch_y = next(iter(train_data))
for i in range(batch_size):
    image_show(batch_X[i], 'ant' if batch_y[i].item() == 0 else 'bee')


def transfer_learning(should_fine_tune_all_layers=False):
    """
    When fine-tuning a model, we load the pretrained model weights and retrain the model with a very small learning rate.
    Otherwise, we freeze the weights of the pretrained model and only train the final layer weights.
    :param should_fine_tune_all_layers:
    :return:
    """

    # Details of models and pretrained weights can be found here: https://pytorch.org/vision/stable/models.html

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if not should_fine_tune_all_layers:  # Freeze the weights of the pretrained model and only train the final layer weights.
        for name, param in resnet.named_parameters():
            param.requires_grad = False

    num_inputs_fc = resnet.fc.in_features
    num_classes = 2  # ants and bees

    resnet.fc = torch.nn.Linear(num_inputs_fc, num_classes)  # requires_grad=True by default

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    training_loop_with_dataloader(resnet, criterion, optimizer, train_data, device=device, epochs=30,
                                  accuracy_fn=cnn_utils.calculate_multi_class_accuracy)

    for _ in range(10):
        test_single_with_dataloader(resnet, val_data, device=device, classification='multiclass',
                                    labels={
                                        0: 'ant',
                                        1: 'bee'})


transfer_learning(should_fine_tune_all_layers=True)