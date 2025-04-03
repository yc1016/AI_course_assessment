import pandas as pd
import os
import csv


def calculate_averages(model_name, image_size, root_path):
    metrics = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Epoch Time', 'Epoch Memory']
    all_metrics_sum = {metric: 0 for metric in metrics}
    num_folds = 5

    for fold in range(num_folds):
        file_name = root_path + f"{model_name}_{image_size}_Fold{fold+1}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            epoch_10_data = df.iloc[9]
            for metric in metrics:
                all_metrics_sum[metric] += epoch_10_data[metric]
        else:
            print(f"File {file_name} not found.")

    averages = {metric: all_metrics_sum[metric] / num_folds for metric in metrics}
    return averages


model_names = ['CNN', 'ViT']
image_sizes = ['64', '128', '224']

csv_data = []
csv_header = ["Model", "Image Size", "Test Accuracy", "Test Loss", "Training Time", "Memory Usage"]
csv_data.append(csv_header)

for model_name in model_names:
    for image_size in image_sizes:
        root_path = f"training_data/{model_name}/"
        averages = calculate_averages(model_name, image_size, root_path)
        
        # calculate the whole traning tiem
        averages['Epoch Time'] *= 10
        csv_data.append([
            model_name,
            image_size,
            f"{averages['Test Accuracy']:.3f}",
            f"{averages['Test Loss']:.3f}",
            f"{averages['Epoch Time']:.3f}",
            f"{averages['Epoch Memory']:.3f}"
        ])
        # Output the avg results
        for metric, avg in averages.items():
            print(f"Average {metric} for {model_name} with {image_size}: {avg}")
        print("-" * 50)


with open('average_training_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print("Results have been saved to average_training_data.csv")
    
