import torch
import torch.nn as nn
from thop import profile


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 15)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.conv_block2(x)
        x = self.pool2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_sizes = [64, 128, 224]
    model_name = "CNN"

    for image_size in image_sizes:
        model = CNN().to(device)
        input_tensor = torch.randn(1, 3, image_size, image_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,))
        print(f"FLOPs of {model_name} for {image_size}x{image_size}: {flops}")
        print(f"Params of {model_name} for {image_size}x{image_size}: {params}")