import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集定义
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

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
dataset = CustomImageDataset(root_dir="data/data", transform=transform)

# CNN 网络定义
class CNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=15):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # 计算展平后的大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
            flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# 训练和测试函数
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 评估
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(val_loader), 100 * correct / total

# 5折交叉验证
# def cross_validate(dataset, k=5, epochs=10, batch_size=64):
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#     fold_results = []

#     for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
#         print(f"\nFold {fold+1}/{k}")

#         train_subset = Subset(dataset, train_idx)
#         val_subset = Subset(dataset, val_idx)

#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#         # 初始化模型、损失函数和优化器
#         model = CNN().to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.0002)

#         # 训练并评估
#         train_losses, train_accuracies, val_losses, val_accuracies = train_and_evaluate(
#             model, train_loader, val_loader, criterion, optimizer, epochs
#         )

#         fold_results.append({
#             "train_losses": train_losses,
#             "train_accuracies": train_accuracies,
#             "val_losses": val_losses,
#             "val_accuracies": val_accuracies
#         })

#     # 计算平均结果
#     avg_train_acc = sum([x["train_accuracies"][-1] for x in fold_results]) / k
#     avg_val_acc = sum([x["val_accuracies"][-1] for x in fold_results]) / k
#     print(f"\nAverage Train Accuracy: {avg_train_acc:.2f}%")
#     print(f"Average Validation Accuracy: {avg_val_acc:.2f}%")

#     return fold_results
def cross_validate(dataset, k=5, epochs=10, batch_size=64):
    labels = [label for _, label in dataset]  # 获取所有样本的标签
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
        print(f"\nFold {fold+1}/{k}")
        
        # train_labels = [labels[idx] for idx in val_idx]
        # train_label_counts = Counter(train_labels)
        # print(f"Training set class distribution: {dict(train_label_counts)}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        # 训练并评估
        train_losses, train_accuracies, val_losses, val_accuracies = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, epochs
        )

        fold_results.append({
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        })

    avg_train_acc = sum([x["train_accuracies"][-1] for x in fold_results]) / k
    avg_val_acc = sum([x["val_accuracies"][-1] for x in fold_results]) / k
    print(f"\nAverage Train Accuracy: {avg_train_acc:.2f}%")
    print(f"Average Validation Accuracy: {avg_val_acc:.2f}%")

    return fold_results

# 运行5折交叉验证
cross_validate(dataset, k=5, epochs=10, batch_size=64)
