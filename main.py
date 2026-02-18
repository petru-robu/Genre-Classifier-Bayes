import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from normalizer import StandardNormalizer
from gnb import GaussianNB
from dnb import DiscreteNB
from hnb import HybridNB


def evaluate_model(model, X_test, y_test):
    """ 
    Evaluates a given model and return accuracy, confusion matrix and per-genre acc.
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)

    classes = np.unique(y_test)
    per_genre_accuracy = {}
    for i, cl in enumerate(classes):
        correct = cm[i, i]
        total = cm[i, :].sum()
        per_genre_accuracy[cl] = correct / total

    return accuracy, cm, per_genre_accuracy


def plot_per_genre_accuracy(per_genre_acc_dicts, labels):
    """
    Plots per-genre accuracy comparison for multiple models.
    """
    classes = list(per_genre_acc_dicts[0].keys())
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, per_genre_acc in enumerate(per_genre_acc_dicts):
        acc_values = [per_genre_acc[cl] * 100 for cl in classes]
        ax.bar(x + i * width, acc_values, width, label=labels[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Genre Accuracy Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/pergenre.svg')


def plot_overall_accuracy(overall_accuracies, labels):
    """
    Plots overall accuracy for multiple models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [acc * 100 for acc in overall_accuracies],
           color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy Comparison")
    plt.savefig('plots/overall.svg')


def plot_confusion_matrices(cms, labels):
    """
    Plots heatmaps of confusion matrices side by side.
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for ax, cm, label in zip(axes, cms, labels):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{label} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.savefig('plots/confusion.svg')



def main():
    # Load data
    df = pd.read_csv("data/training-data.csv")
    non_feature_cols = ["filename", "label",
                        "tempo", "harmony_mean", "harmony_var"]

    X = df.drop(columns=non_feature_cols).values
    y = df["label"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y
    )

    # Normalize
    normalizer = StandardNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    # Initialize models
    models = [
        ("Gaussian NB", GaussianNB()),
        ("Discrete NB", DiscreteNB(n_bins=10, smoothing=1.0)),
        ("Hybrid NB", HybridNB(n_bins=10, smoothing=1.0, skew_threshold=1.0))
    ]

    overall_accuracies = []
    per_genre_acc_dicts = []
    confusion_matrices = []

    for name, model in models:
        print(f"----{name}----")
        model.fit(X_train_norm, y_train)
        accuracy, cm, per_genre_accuracy = evaluate_model(
            model, X_test_norm, y_test)
        overall_accuracies.append(accuracy)
        per_genre_acc_dicts.append(per_genre_accuracy)
        confusion_matrices.append(cm)

        print(f"Accuracy: {accuracy*100:.2f}%")
        print("Confusion matrix:")
        print(cm)
        print("\nPer-genre accuracy:")
        for cl, acc in per_genre_accuracy.items():
            print(f"{cl}: {acc*100:.2f}%")
        print("\n")

    plot_per_genre_accuracy(per_genre_acc_dicts, [name for name, _ in models])

    plot_overall_accuracy(overall_accuracies, [name for name, _ in models])

    plot_confusion_matrices(confusion_matrices, [name for name, _ in models])


if __name__ == "__main__":
    main()
