import torch.nn as nn
import torch.optim as optim

class Linear(nn.Module):
    def __init__(self, epochs):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.epochs = epochs
        self.channels = 1

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Преобразуем изображение в вектор
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Convolutional(nn.Module):
    def __init__(self, epochs, channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(channels, 32, (3, 3))
        self.conv_2 = nn.Conv2d(32, 64, (3, 3))
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.layer_out = nn.Linear(64 * 24 * 24, 4)
        self.epochs = epochs
        self.channels = channels

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.relu(x)
        out = self.layer_out(x)

        return out

def get_optimizer(model: nn.Module):
    return optim.Adam(model.parameters())


def get_criterion():
    return nn.CrossEntropyLoss()


def create_linear_model(epochs):
    return Linear(epochs)


def create_convolutional_model(epochs, channels):
    return Convolutional(epochs, channels)
