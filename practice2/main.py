import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Преобразуем изображение в вектор
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


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
        if (epoch + 1) % 10 == 0: print(f"Поколение {epoch + 1}/{num_epochs}, Ошибка: {avg_loss:.6f}")
    ax.plot(loss_x, loss_y)
    plt.show()

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

def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    print(f"Предсказанный класс: {class_names[predicted.item()]}")
    return predicted.item()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: ImageOps.invert(x)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = "./Lowercase"
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    class_names = dataset.classes

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DigitRecognizer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, criterion, optimizer, train_loader, num_epochs=150)

    evaluate_model(model, test_loader)

    custom_image_path = "./Lowercase/Epsilon/4.jpg"
    predict_image(model, custom_image_path, transform, class_names)

    # torch.save(model.state_dict(), "1500.pth")
