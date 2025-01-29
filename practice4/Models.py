import torch.nn as nn
from torch import optim

class TemperatureRNN(nn.Module):
    def __init__(self, sequence_length, input_size=1,
                 hidden_size=64, output_size=1, num_layers=1):
        super(TemperatureRNN, self).__init__()
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_hidden_state = rnn_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

    @staticmethod
    def get_criterion():
        return nn.MSELoss()

    def get_optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)