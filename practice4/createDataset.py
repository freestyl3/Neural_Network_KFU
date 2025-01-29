import torch

from Dataset import TemperatureDataset

def create_dataset(temperature_data, sequence_length):
    dataset = TemperatureDataset(temperature_data, sequence_length)
    data = dataset.generate_data()

    train_data = data[:int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader