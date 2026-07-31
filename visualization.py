from typing import Dict

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay


def plot_confusion_matrix(y_true: list, y_pred: list, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_roc_curve(y_true: list, y_scores: list, title: str):
    RocCurveDisplay.from_predictions(y_true, y_scores, name=title)
    plt.title(title)
    plt.show()


def visualize_results(results: Dict[str, Dict[str, float]]):
    # Plot bar chart comparing classifier metrics
    classifiers = list(results.keys())
    metrics = ['precision', 'recall', 'f1']

    fig, ax = plt.subplots()
    for i, metric in enumerate(metrics):
        values = [results[clf][metric] for clf in classifiers]
        ax.bar([x + i*0.2 for x in range(len(classifiers))], values, width=0.2, label=metric)

    ax.set_xticks([x + 0.2 for x in range(len(classifiers))])
    ax.set_xticklabels(classifiers)
    ax.legend()
    plt.title('Classifier Performance Comparison')
    plt.show()
