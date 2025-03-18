import torch
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for file in os.listdir(root_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 过滤图像文件
                label = int(file.split('_')[-1].split('.')[0])  # 提取文件名中的标签
                self.image_files.append(file)
                self.labels.append(label-1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

# 创建数据集
dataset = CustomImageDataset(root_dir="data/data", transform=transform)

# 划分训练和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# DataLoader for batching and shuffling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model configuration with increased capacity
config = ViTConfig(
    image_size=224,
    num_channels=3,  # Grayscale input for MNIST
    num_labels=15,   # 10 classes for MNIST
    hidden_size=256,  # Larger hidden size
    num_hidden_layers=8,  # Increased number of layers
    num_attention_heads=8,
    intermediate_size=512  # Larger intermediate size
)
model = ViTForImageClassification(config).to(device)
# model = ViTForImageClassification.from_pretrained("pretrained_ViT/", num_labels=15, ignore_mismatched_sizes=True).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# Training loop with accuracy tracking
def train_vit_with_accuracy(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        # Training step
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        # Track and log training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy after each epoch
        test_loss, test_accuracy = evaluate_accuracy(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    # Plot metrics after training
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test accuracy calculation
def evaluate_accuracy(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Testing loop with confidence display
def test_vit_with_confidence(model, test_loader):
    model.eval()
    correct_images = []
    incorrect_images = []
    correct_preds = []
    incorrect_preds = []

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probs = torch.softmax(outputs, dim=1)  # Compute softmax probabilities
            conf, predicted = torch.max(probs, 1)

            for i in range(images.size(0)):
                if predicted[i] == labels[i]:
                    correct_images.append(images[i].cpu())
                    correct_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))
                else:
                    incorrect_images.append(images[i].cpu())
                    incorrect_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))

    # Visualize correct and incorrect predictions
    show_images(correct_images, correct_preds, "Correct Predictions (with Confidence)", max_images=5)
    show_images(incorrect_images, incorrect_preds, "Incorrect Predictions (with Confidence)", max_images=5)

# Visualization helper function
def show_images(images, preds_labels, title, max_images=10, images_per_row=5):
    rows = (max_images + images_per_row - 1) // images_per_row
    plt.figure(figsize=(15, rows * 3))
    for i in range(min(max_images, len(images))):
        plt.subplot(rows, images_per_row, i + 1)
        img = images[i].squeeze(0).numpy()
        pred, label, conf = preds_labels[i]
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}\nPred: {pred} ({conf:.2f})")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Train and Test the model
train_vit_with_accuracy(model, train_loader, test_loader, criterion, optimizer, epochs=10)
test_vit_with_confidence(model, test_loader)