import torch

class TemperatureDataset:
    def __init__(self, temperature_data, sequence_length):
        self.temperature_data = temperature_data
        self.sequence_length = sequence_length

    def generate_data(self):
        data = []
        for i in range(len(self.temperature_data) - self.sequence_length):
            sequence = self.temperature_data[i:i + self.sequence_length]
            label = self.temperature_data[i + self.sequence_length]  # Следующая температура
            data.append((torch.tensor(sequence, dtype=torch.float32), 
                         torch.tensor(label, dtype=torch.float32)))
        return data