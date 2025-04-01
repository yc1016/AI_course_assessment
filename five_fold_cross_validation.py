import os
import csv
import time
import torch
import psutil
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from thop import profile
from models.CNN import CNN
from models.ViT import ViT

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

image_sizes = [64, 128, 224]  # the three image sizes for experiment
model_names = ['CNN', 'ViT']
labels_folder = "labels_data/"
training_data_folder = "training_data/"
target_classes = {"CNN": [9, 12], "ViT": [3, 12]} # the classes observed in each model

epochs = 10
best_lr = 0.0001
best_bs = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for file in os.listdir(root_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                mark = int(file.split('_')[1])
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

'''get_model():
   Get CNN or ViT model from models folder
'''
def get_model(model_name, image_size):
    if model_name == "ViT":
        return ViT(image_size).to(device)
    else:
        return CNN().to(device)


'''show_distribution():
   Print the distribution of classes in training set and test set
'''
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


'''show_images():
   Test some images in test set with confidence
'''
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


'''show_flops_params():
   Calculate the FLOPs and Params of the model by using profile() function
'''
def show_flops_params(model_name, image_size):
    model = get_model(model_name, image_size)
    input_tensor = torch.randn(1, 3, image_size, image_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs of {model_name} for {image_size}x{image_size}: {flops}")
    print(f"Params of {model_name} for {image_size}x{image_size}: {params}")


'''get_memory_usage():
   Calculate the memory usage
'''
def get_memory_usage():
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)  # change to (MB)
    else:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 ** 2)  # change to (MB)


'''save_model():
   Save the weights of the trained model to weights folder
'''
def save_model(model, model_name, image_size, fold):
    model_dir = "weights"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_{image_size}_Fold{fold+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


'''train_model():
   Train the model and print some metrics
     (train_loss, train_acc, test_loss, test_acc, time, memory)
   during training process
'''
def train_model(fold, model_name, model, train_loader, test_loader, criterion, optimizer, epochs=10):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    test_losses = []
    epoch_times = []
    epoch_memory_usages = []
    save_folder = f"{training_data_folder}{model_name}/"

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") # using tqdm() to show progress

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

        test_loss, test_accuracy = evaluate_model(model_name, model, test_loader, criterion, fold, epoch+1)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # calculate the time spent in this epoch
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        # get the memory usage during training
        memory_usage = get_memory_usage()
        epoch_memory_usages.append(memory_usage)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s, Memory: {memory_usage:.2f}MB")
    
    # Save the model in each fold validation
    save_model(model, model_name, image_size, fold)

    # Save the metrics data to .csv file
    os.makedirs(save_folder, exist_ok=True) 
    csv_file = save_folder + f"{model_name}_{image_size}_Fold{fold+1}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Epoch Time', 'Epoch Memory'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], train_accuracies[i], test_losses[i], test_accuracies[i], epoch_times[i], epoch_memory_usages[i]])

    return test_accuracies[-1]


'''evaluate_model():
   Evaluate the model by test_acc and test_loss
   Save the true_labels and pred_labels to create confusion matrix
   Give a report for appointed classes by using classfication_report() function
'''
def evaluate_model(model_name, model, test_loader, criterion, fold, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    save_folder = f"{labels_folder}{model_name}/"

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

            # record the true_labels and pred_labels to calculate metrics
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    os.makedirs(save_folder, exist_ok=True)
    # just save the last epoch
    if epoch == 10:
        np.save(save_folder+f"{model_name}_{image_size}_Fold{fold+1}_true_labels.npy", np.array(true_labels))
        np.save(save_folder+f"{model_name}_{image_size}_Fold{fold+1}_pred_labels.npy", np.array(pred_labels))   
        
        target_labels = target_classes[model_name]
        report = classification_report(true_labels, pred_labels, labels=target_labels)
        print(f"Classification Report for {model_name} {image_size} Fold{fold + 1}:\n{report}")

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy



def test_model_with_confidence(model_name, model, test_loader):
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



for image_size in image_sizes:
    for model_name in model_names:
        print(f"===================== Image size: {image_size} Model: {model_name} =====================")

        show_flops_params(model_name, image_size)

        # do some pre-processing
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CustomImageDataset(root_dir="data/data", transform=transform)

        k = 5 
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # using StratifiedKFold() to keep the distribution of each class
        all_accuracies = []
        labels = [label for _, label in dataset]


        # start the five fold cross validation
        for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"Fold {fold + 1}/{k}")
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, val_indices)

            print(f"Training dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")

            show_distribution(test_dataset, train_dataset)

            # learning_rates = [0.0001]
            # batch_sizes = [64]
            # best_lr, best_bs = grid_search(model_name, learning_rates, batch_sizes, epochs, criterion, image_size)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_bs, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_bs, shuffle=False)
            model = get_model(model_name, image_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

            accuracy = train_model(fold, model_name, model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
            all_accuracies.append(accuracy)

        print(f"Average accuracy for image size {image_size}: {np.mean(all_accuracies):.2f}%")
