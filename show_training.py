'''
show_training.py:
    Used to check the accuracy and loss of model during training
'''
import os
import pandas as pd
import matplotlib.pyplot as plt


root_path = "results/"
# data_path = "CNN_64_training_data.csv"
# data_path = "CNN_128_training_data.csv"
# data_path = "CNN_224_training_data.csv"

# data_path = "ViT_64_training_data.csv"
# data_path = "ViT_128_training_data.csv"
data_path = "ViT_224_training_data.csv"

data = pd.read_csv(root_path+data_path)
model_name = data_path.split('_')[0]
image_size = data_path.split('_')[1]+"x"+data_path.split('_')[1]

epochs = data['Epoch']
train_loss = data['Train Loss']
train_accuracy = data['Train Accuracy']
test_loss = data['Test Loss']
test_accuracy = data['Test Accuracy']
epoch_time = data['Epoch Time']


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, test_loss, label='Test Loss', marker='o')
plt.xlabel('Epoch', fontsize=16, fontfamily="Times New Roman")
plt.ylabel('Loss', fontsize=16, fontfamily="Times New Roman")
plt.legend(prop={'size': 12, 'family': 'Times New Roman'})
plt.grid(True)
plt.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.1)
loss_img_path = os.path.join(root_path, f'{model_name}_{image_size}_Loss-Epoch.png')
plt.savefig(loss_img_path)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(epochs, test_accuracy, label='Test Accuracy', marker='o')
plt.xlabel('Epoch', fontsize=16, fontfamily="Times New Roman")
plt.ylabel('Accuracy (%)',fontsize=16, fontfamily="Times New Roman")
plt.legend(prop={'size': 12, 'family': 'Times New Roman'})
plt.grid(True)
plt.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.1)
accuracy_img_path = os.path.join(root_path, f'{model_name}_{image_size}_Accuracy-Epoch.png')
plt.savefig(accuracy_img_path)
plt.show()