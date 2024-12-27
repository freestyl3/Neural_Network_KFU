import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

from Models import (create_linear_model, create_convolutional_model,
                    get_criterion, get_optimizer)
from Dataset import create_dataset, divide_dataset, get_transform
from Randomize_data import randomize_data

# Тренировка модели
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    loss_x = []
    loss_y = []
    fig, ax = plt.subplots()
    ax.set_title("Ошибка на обучении")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Ошибка")

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_x.append(epoch)
        loss_y.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Поколение {epoch + 1}/{num_epochs}, Ошибка: {avg_loss:.6f}")
    ax.plot(loss_x, loss_y)
    plt.show()

# Проверка модели на точность
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Точность модели: {accuracy * 100:.2f}%")
    return accuracy

# Предсказание модели
def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        num, predicted = torch.max(output, 1)
    return predicted.item()


def show_predicts(case, transform, model, class_names):
    for class_name in class_names:
        figure, ax = plt.subplots(3, 4)
        for i in range(10):
            a, b = i % 3, i // 3
            image_path = f"./test_images/{case}/{class_name}/{i}.jpg"
            img = Image.open(image_path)
            img = img.resize((28, 28))
            ax[a, b].imshow(img, cmap="gray")
            predicted_class = class_names[
                predict_image(model, image_path, transform)
            ]
            is_true = class_name == predicted_class
            ax[a, b].set_title(predicted_class,
                               color='green' if is_true else 'red')
    plt.show()


def main():
    case = 'Lowercase'

    # Создание модели линейной нейронной сети
    model = create_linear_model(300)

    # Создание модели сверточной нейронной сети
    # model = create_convolutional_model(100, 3)

    criterion = get_criterion()
    optimizer = get_optimizer(model)

    # Создание датасета
    transform = get_transform(28, model.channels)
    # dataset = create_dataset(f'./{case}', transform)
    # class_names = dataset.classes
    #
    # # Деление датасета на обучающий и тестовый
    # train_dataset, test_dataset = divide_dataset(dataset)

    train_dataset = create_dataset(f"./{case}", transform)
    test_dataset = create_dataset(f'./test_images/{case}', transform)
    class_names = train_dataset.classes

    # Загрузка датасетов
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Тренировка модели
    train_model(model, criterion, optimizer,
                train_loader, num_epochs=model.epochs)

    # Проверка точности модели
    evaluate_model(model, test_loader)

    # Рандомизация объектов
    randomize_data(class_names, f'{case}')

    # Показ предсказаний нейронной сети
    show_predicts(case, transform, model, class_names)

if __name__ == "__main__":
    main()
