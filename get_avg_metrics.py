import csv
import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_metrics(true_labels, pred_labels, num_classes):
    accuracies = np.zeros(num_classes)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)
    specificities = np.zeros(num_classes)

    for i in range(num_classes):
        # the true_labels is i, pred_labels is i
        tp = np.sum((true_labels == i) & (pred_labels == i))
        # the pred_labels and true_labels are equal, true_labels is not i
        tn = np.sum((true_labels == pred_labels) & (true_labels != i))
        # the true_labels is not i, pred_labels is i
        fp = np.sum((true_labels != i) & (pred_labels == i))
        # the true_labels is i, pred_labels is not i
        fn = np.sum((true_labels == i) & (pred_labels != i))

        accuracies[i] = tp / np.sum(true_labels == i)

        if tp + fp > 0:
            precisions[i] = tp / (tp + fp)
            
        if tp + fn > 0:
            recalls[i] = tp / (tp + fn)

        if precisions[i] + recalls[i] > 0:
            f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        
        if tn + fp > 0:
            specificities[i] = tn / (tn + fp)

    return accuracies, precisions, recalls, f1_scores, specificities


def calculate_average_metrics(image_size, model_name, num_classes, root_path):
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_specificities = []

    for fold in range(5):
        true_labels_path = root_path + f"{model_name}_{image_size}_Fold{fold + 1}_true_labels.npy"
        pred_labels_path = root_path + f"{model_name}_{image_size}_Fold{fold + 1}_pred_labels.npy"
        true_labels = np.load(true_labels_path)
        pred_labels = np.load(pred_labels_path)

        accuracies, precisions, recalls, f1_scores, specificities = calculate_metrics(true_labels, pred_labels, num_classes)

        all_accuracies.append(accuracies)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_f1_scores.append(f1_scores)
        all_specificities.append(specificities)

    avg_accuracies = np.mean(all_accuracies, axis=0)
    avg_precisions = np.mean(all_precisions, axis=0)
    avg_recalls = np.mean(all_recalls, axis=0)
    avg_f1_scores = np.mean(all_f1_scores, axis=0)
    avg_specificities = np.mean(all_specificities, axis=0)


    return avg_accuracies, avg_precisions, avg_recalls, avg_f1_scores, avg_specificities

num_classes = 15
image_sizes = [64, 128, 224]
model_names = ["CNN", "ViT"]
target_classes = {"CNN": [9, 12], "ViT": [3, 12]} 

# Prepare data for CSV
csv_data = []
csv_header = ["Model", "Image Size", "Class", "Average Accuracy", "Average Precision", "Average Recall", "Average F1 Score", "Average Specificity"]
csv_data.append(csv_header)

for model_name in model_names:
    for image_size in image_sizes:
        root_path = f"labels_data/{model_name}/"
        print(f"Model: {model_name}, Image Size: {image_size}")
        avg_accuracies, avg_precisions, avg_recalls, avg_f1_scores, avg_specificities = calculate_average_metrics(
            image_size, model_name, num_classes, root_path)

        if avg_accuracies is not None:
            # Output the results for specified classes, rounded to three decimal places
            for cls in target_classes[model_name]:
                print(f"Average Accuracy for Class {cls}: {avg_accuracies[cls]:.3f}")
                print(f"Average Precision for Class {cls}: {avg_precisions[cls]:.3f}")
                print(f"Average Recall for Class {cls}: {avg_recalls[cls]:.3f}")
                print(f"Average F1 Score for Class {cls}: {avg_f1_scores[cls]:.3f}")
                print(f"Average Specificity for Class {cls}: {avg_specificities[cls]:.3f}")
                print()

                # Add data to CSV
                csv_data.append([model_name, image_size, cls, f"{avg_accuracies[cls]:.3f}", f"{avg_precisions[cls]:.3f}",
                                 f"{avg_recalls[cls]:.3f}", f"{avg_f1_scores[cls]:.3f}", f"{avg_specificities[cls]:.3f}"])
        print("-" * 50)

# Save data to CSV
with open('classification_metrics_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print("Results have been saved to classification_metrics_results.csv")