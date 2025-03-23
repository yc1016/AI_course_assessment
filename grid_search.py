import torch
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import csv
import time


image_size = 64
model_name = "ViT"
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


class CNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=15):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(0.25)

        # 计算展平后的大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
            flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        # self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        # x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.dropout2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # x = self.dropout3(x)
        x = self.fc2(x)
        return x


def get_model(model_name):
    if model_name == "ViT":
        config = ViTConfig(
            image_size=image_size,
            patch_size=8,
            num_channels=3,
            num_labels=15,
            hidden_size=256,
            num_hidden_layers=10,
            num_attention_heads=8,
            intermediate_size=512
        )
        return ViTForImageClassification(config).to(device)
    else:
        return CNN((3, image_size, image_size)).to(device)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for file in os.listdir(root_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                mark = int(file.split('_')[1])
                # if mark > 40: 
                #     continue
                # if (int(file.split('_')[-1].split('.')[0]) >= 12):
                #     continue
                label = int(file.split('_')[-1].split('.')[0])
                self.image_files.append(file)
                self.labels.append(label - 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def show_distribution(test_dataset, train_dataset):
    train_class_counts = {}
    for _, label in train_dataset:
        if label not in train_class_counts:
            train_class_counts[label] = 0
        train_class_counts[label] += 1

    test_class_counts = {}
    for _, label in test_dataset:
        if label not in test_class_counts:
            test_class_counts[label] = 0
        test_class_counts[label] += 1

    print("Training set class counts:")
    for class_label, count in sorted(train_class_counts.items()):
        print(f"Class {class_label}: {count}")

    print("Test set class counts:")
    for class_label, count in sorted(test_class_counts.items()):
        print(f"Class {class_label}: {count}")


def show_images(images, preds_labels, title, max_images=10, images_per_row=5):
    rows = (max_images + images_per_row - 1) // images_per_row
    plt.figure(figsize=(15, rows * 3))
    for i in range(min(max_images, len(images))):
        plt.subplot(rows, images_per_row, i + 1)
        img = images[i].squeeze(0).permute(1, 2, 0).numpy()
        pred, label, conf = preds_labels[i]
        plt.imshow(img)
        plt.title(f"Label: {label}\nPred: {pred} ({conf:.2f})")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def train_vit_with_accuracy(model_name, model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if model_name == "ViT":
                outputs = model(images).logits
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = evaluate_accuracy(model_name, model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s")
        # if epoch > 2 and test_accuracy < 40:
        #     break

    csv_file = f"{model_name}_{image_size}_training_data.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Epoch Time'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], train_accuracies[i], test_losses[i], test_accuracies[i], epoch_times[i]])

    return test_accuracies[-1]


def evaluate_accuracy(model_name, model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if model_name == "ViT":
                outputs = model(images).logits
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test_vit_with_confidence(model_name, model, test_loader):
    model.eval()
    correct_images = []
    incorrect_images = []
    correct_preds = []
    incorrect_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if model_name == "ViT":
                outputs = model(images).logits
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

            for i in range(images.size(0)):
                if predicted[i] == labels[i]:
                    correct_images.append(images[i].cpu())
                    correct_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))
                else:
                    incorrect_images.append(images[i].cpu())
                    incorrect_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))

    show_images(correct_images, correct_preds, "Correct Predictions (with Confidence)", max_images=5)
    show_images(incorrect_images, incorrect_preds, "Incorrect Predictions (with Confidence)", max_images=5)


def grid_search(model_name, learning_rates, batch_sizes, epochs, criterion):
    results = {}
    best_accuracy = 0
    best_lr = None
    best_bs = None

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"========Testing learning rate: {lr}, batch size: {bs}==========")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
            model = get_model(model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            accuracy = train_vit_with_accuracy(model_name, model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
            results[(lr, bs)] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
                best_bs = bs

    print(f"All results: {results}")
    print(f"Best learning rate: {best_lr}, best batch size: {best_bs}, best accuracy: {best_accuracy:.2f}%")
    return best_lr, best_bs


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = CustomImageDataset(root_dir="data/data", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

show_distribution(test_dataset, train_dataset)
# exit()

'''ViT-64x64: lr = 0.0001 bs = 64'''
'''ViT-128x128: lr = 0.0001 bs = 64'''
'''ViT-224x224: lr = 0.0001 bs = 64'''

'''CNN-64x64: lr = 0.0001 bs = 64'''
'''CNN-128x128: lr = 0.0001 bs = 64'''
'''CNN-224x224: lr = 0.0001 bs = 64'''
learning_rates = [0.0001]
batch_sizes = [64]

best_lr, best_bs = grid_search(model_name, learning_rates, batch_sizes, epochs, criterion)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_bs, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_bs, shuffle=False)
# model = get_model(model_name)
# optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
# train_vit_with_accuracy(model_name, model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
# test_vit_with_confidence(model_name, model, test_loader)
    