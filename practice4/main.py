import json

import torch
import matplotlib.pyplot as plt

from collectData import collect_data_by_years
from createDataset import create_dataset
from Models import TemperatureRNN

SEQUENCE_LENGTH = 3

def get_data(path):
    collect_data_by_years(2023, 2024)

    with open(path, 'r') as file:
        data = json.load(file)
    # print(json.dumps(data, indent=4))

    # print([hourly_data['temp'] for hourly_data in data['data']])
    return [hourly_data['temp'] for hourly_data in data['data']]


def train_model(model, path):
    criterion = TemperatureRNN.get_criterion()
    optimizer = model.get_optimizer()

    train_loader, test_loader = create_dataset(get_data(path), SEQUENCE_LENGTH)

    epochs = [i for i in range(101)]
    train_loss = []
    tests_loss = []

    for epoch in epochs:
        model.train()
        epoch_loss = 0.0
        for seq, label in train_loader:
            seq = seq.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(seq)

            loss = criterion(output, label.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        train_loss.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for seq, label in test_loader:
                seq = seq.unsqueeze(-1)
                output = model(seq)
                loss = criterion(output, label.unsqueeze(-1))
                test_loss += loss.item()
            tests_loss.append(test_loss / len(test_loader))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs[-1]}, "
                  f"Loss: {(epoch_loss / len(train_loader)):.4f}, "
                  f"Test Loss: {(test_loss / len(test_loader)):.4f}")

    plt.figure(figsize=(15, 9))
    plt.xlabel("Поколения")
    plt.ylabel("Ошибка")
    plt.title("Ошибка на обучении и тестировании модели")
    plt.plot(epochs, train_loss, color='r')
    plt.plot(epochs, tests_loss, color='b')
    plt.legend(('Ошибка на обучении', 'Ошибка на тестировании'))
    plt.show()

def predict(model, path):
    model.eval()

    with open(path, 'r') as file:
        data = json.load(file)
    temperature_data = []
    for hourly_data in data['data']:
        temperature_data.append(hourly_data['temp'])

    time_data = [i for i in range(len(temperature_data) - SEQUENCE_LENGTH)]
    predicted_values = []
    true_values = temperature_data[SEQUENCE_LENGTH - 1:len(temperature_data) - 1]

    for i in range(len(temperature_data) - SEQUENCE_LENGTH):
        with torch.no_grad():
            last_sequence = torch.tensor(temperature_data[i:i + SEQUENCE_LENGTH], dtype=torch.float32).unsqueeze(
                0).unsqueeze(-1)
            predicted_temperature = model(last_sequence).item()
            predicted_values.append(predicted_temperature)

    plt.figure(figsize=(15, 9))
    plt.title("Предсказание модели на месяц")
    plt.xlabel("Час")
    plt.ylabel("Температура")
    plt.plot(time_data, true_values, color='b')
    plt.plot(time_data, predicted_values, color='r')
    plt.legend(('Точное значение', 'Предсказанное значение'))
    plt.show()
    plt.figure(figsize=(15, 9))
    plt.title("Предсказание модели на первую неделю")
    plt.xlabel("Час")
    plt.ylabel("Температура")
    plt.plot(time_data[:168], true_values[:168], color='b')
    plt.plot(time_data[:168], predicted_values[:168], color='r')
    plt.legend(('Точное значение', 'Предсказанное значение'))
    plt.show()

if __name__ == '__main__':
    model = TemperatureRNN(SEQUENCE_LENGTH, input_size=1,
                           hidden_size=64, output_size=1)
    train_model(model, './Data/Moscow/January2023.json')
    predict(model, './Data/Kazan/January2024.json')
