import pandas as pd
from PIL import Image
import cv2 
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import tensorflow as tf
from functools import partial
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 加载数据集与标签
dataframe = pd.read_csv('data/chinese_mnist.csv')
index = dataframe.iloc[:,:-2].values 
value = dataframe["code"].values
chineseNumber = dataframe["character"]

# print(index.shape)
# print(len(value))

filename_label_link = {}

filename_list = []
label_list = []
X = []
y = [] 
for i in range(0,len(value)):
    x = index[i]
    filename = "input_%s_%s_%s.jpg" % (x[0], x[1], x[2])
    val = value[i] 
    filename_label_link[filename] = val
    filename_list.append(filename)
    label_list.append(val)
    a = cv2.imread("data/data/%s"%filename)
    X.append(a)
    y.append(val)


class CustomImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        # 将 numpy.ndarray 转换为 PIL.Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # 如果不是 uint8 类型，需要先转换为 uint8
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# 分割数据集
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train[:, 1:]
y_test = y_test[:, 1:]

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.RandomHorizontalFlip(p=0.1),  # Data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize images to [-1, 1]
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize images to [-1, 1]
])

train_dataset = CustomImageDataset(X_train, y_train, transform=transform1)
test_dataset = CustomImageDataset(X_test, y_test, transform=transform2)

batch_size = 64  # 可以根据需要调整批量大小
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 打印一些信息进行验证
# print("训练集 DataLoader 长度:", len(train_dataloader))
# print("测试集 DataLoader 长度:", len(test_dataloader))

def test_images(num_images, dataloader):
    images, labels = next(iter(dataloader))

    images = images * 0.5 + 0.5

    images = images.numpy()
    labels = labels.numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')

    plt.show()

# test_images(3, train_dataloader)
# exit()
def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # print("Input data shape:", data.shape)  # 添加打印语句查看数据形状
        # exit()
        data, target = data.to(device), target.to(device)
        target = torch.argmax(target, dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy_float = correct / len(data)
        loss_float = loss.item()
        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )

    return loss_history, accuracy_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        target = torch.argmax(target, dim=1)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct / len(val_loader.dataset)


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        target = torch.argmax(target, dim=1)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)

        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points


def run_training(
    model,
    num_epochs,
    optimizer_kwargs,
    train_loader,
    val_loader,
    device="cuda",
):
    # ===== Data Loading =====

    # ===== Model, Optimizer and Criterion =====
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 3, 1)
    plt.plot(t_train, train_acc_history, label="Train")
    plt.plot(t_val, val_acc_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 2)
    plt.plot(t_train, train_loss_history, label="Train")
    plt.plot(t_val, val_loss_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(t_train, lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    # ===== Plot low/high loss predictions on validation set =====
    points = get_predictions(
        model,
        device,
        val_loader,
        partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1])
    plt.figure(figsize=(15, 6))
    for k in range(5):
        plt.subplot(2, 5, k + 1)
        plt.imshow(points[k][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        plt.subplot(2, 5, 5 + k + 1)
        plt.imshow(points[-k - 1][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")

    return sum(train_acc) / len(train_acc), val_acc

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
                
        convs = [
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, dilation=2),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, dilation=2),
            nn.MaxPool2d(2),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, dilation=2),
            #nn.Conv2d(512, 1024, kernel_size=2, stride=2, dilation=2),
            nn.MaxPool2d(2),
            nn.GELU()
        ]
        
        self.conv = nn.Sequential(*convs)
        
        self.linear = nn.Linear(2048, 15)
        
        self.log_softmax = nn.LogSoftmax(dim=0) # avoid overflowing: large number -> exp() -> NaN -> log() -> NaN. I think I could also solve this through batch normalization.
        
    def forward(self, x):
        # x = x.unsqueeze(1) # single channel image
        #print(x.size())
        
        hidden = self.conv(x)
        #print(hidden.size())
        
        hidden = torch.flatten(hidden, start_dim=1)
        #print(hidden.size())
        
        
        output = self.log_softmax(self.linear(hidden))
        return output

config = ViTConfig(
    image_size=224,
    num_channels=3,
    num_labels=15,   
    hidden_size=256,  # Larger hidden size
    num_hidden_layers=8,  # Increased number of layers
    num_attention_heads=8,
    intermediate_size=512  # Larger intermediate size
)
# model_factory = ViTForImageClassification(config)
# model_factory = ViTForImageClassification.from_pretrained(
#     './pretrained_ViT',
#     config=config,
#     ignore_mismatched_sizes=True
# )
model_factory = ConvNet()
image_size = 224
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer_kwargs = dict(
    lr=0.001,
    # weight_decay=1e-2,
)

run_training(
    model=model_factory,
    num_epochs=num_epochs,
    optimizer_kwargs=optimizer_kwargs,
    train_loader=train_dataloader,
    val_loader=test_dataloader,
    device=device,
)
