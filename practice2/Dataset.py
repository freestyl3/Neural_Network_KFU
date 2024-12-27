import torch
from torchvision import datasets, transforms
from PIL import ImageOps

# Функция трансформация изображений в вектора 28х28
def get_transform(size, output_channels):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=output_channels),
        transforms.Resize((size, size)),
        transforms.Lambda(lambda x: ImageOps.invert(x)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Создание датасета
def create_dataset(path, transform):
    return datasets.ImageFolder(root=path, transform=transform)

# Деление датасета
def divide_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])

