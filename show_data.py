import os
import pandas as pd
import matplotlib.pyplot as plt


root_path = "images/"
# data_path = "CNN_64_training_data.csv"
# data_path = "CNN_128_training_data.csv"
# data_path = "CNN_224_training_data.csv"

# data_path = "ViT_64_training_data.csv"
# data_path = "ViT_128_training_data.csv"
data_path = "ViT_224_training_data.csv"
# 读取数据
data = pd.read_csv(root_path+data_path)
model_name = data_path.split('_')[0]
image_size = data_path.split('_')[1]+"x"+data_path.split('_')[1]

# 提取数据
epochs = data['Epoch']
train_loss = data['Train Loss']
train_accuracy = data['Train Accuracy']
test_loss = data['Test Loss']
test_accuracy = data['Test Accuracy']
epoch_time = data['Epoch Time']

# 绘制训练损失和测试损失随Epoch的变化
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, test_loss, label='Test Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{model_name}_{image_size}_Loss-Epoch')
plt.legend()
plt.grid(True)
loss_img_path = os.path.join(root_path, f'{model_name}_{image_size}_Loss-Epoch.png')
plt.savefig(loss_img_path)
plt.show()

# 绘制训练准确率和测试准确率随Epoch的变化
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'{model_name}_{image_size}_Accuracy-Epoch')
plt.legend()
plt.grid(True)
accuracy_img_path = os.path.join(root_path, f'{model_name}_{image_size}_Accuracy-Epoch.png')
plt.savefig(accuracy_img_path)
plt.show()

# 绘制每轮训练时间随Epoch的变化
plt.figure(figsize=(12, 6))
plt.bar(epochs, epoch_time, color='g', width=0.5)
plt.xlabel('Epoch')
plt.ylabel('Epoch Time (s)')
plt.title(f'{model_name}_{image_size}_Each epoch time')
plt.grid(True)
plt.show()