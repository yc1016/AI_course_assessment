'''
get_confusion_matrix.py:
    Used to draw the confusion matrix by the true_labels and pred_labels stored in labels_data folder
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(true_labels, pred_labels, class_names, model_name, image_size, fold):
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction label')
    plt.ylabel('Actual label')
    plt.savefig(f"{model_name}_{image_size}_Fold{fold}confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.show()

root_path = "labels_data/"
model_name = "CNN"
image_size = 224
fold = 1
true_labels = np.load(root_path + f"{model_name}_{image_size}_Fold{fold}_true_labels.npy") 
pred_labels = np.load(root_path + f"{model_name}_{image_size}_Fold{fold}_pred_labels.npy") 
class_names = [str(i) for i in range(15)] 
plot_confusion_matrix(true_labels, pred_labels, class_names, model_name, image_size, fold)
