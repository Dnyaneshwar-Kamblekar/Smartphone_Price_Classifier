
"""Evaluation metrics and plots."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from .preprocessing import CLASS_NAMES


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model and return its performance metrics."""

    predictions = model.predict(X_test)

    labels = [0, 1, 2, 3]
    target_names = [CLASS_NAMES[i] for i in labels]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    return {
        "model": name,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        ),
    }


def save_evaluation(results, reports_dir: Path):
    """Save evaluation metrics, comparison charts and CSV reports."""

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics as JSON
    payload = {
        "dataset": "Kaggle Mobile Price Classification",
        "test_size": 0.20,
        "random_state": 42,
        "averaging": "weighted",
        "models": results,
    }

    metrics_path = reports_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    if not results:
        raise ValueError(
            "No model evaluation results were provided."
        )

    # Model accuracy comparison chart
    names = [result["model"] for result in results]
    scores = [result["accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(names, scores)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.grid(axis="y", alpha=0.25)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 0.015, 0.98),
            f"{score:.3f}",
            ha="center"
        )

    plt.setp(
        ax.get_xticklabels(),
        rotation=15,
        ha="right"
    )

    fig.tight_layout()

    fig.savefig(
        reports_dir / "model_comparison.png",
        dpi=150
    )

    plt.close(fig)

    # Confusion matrix charts
    number_of_models = len(results)

    fig, axes = plt.subplots(
        1,
        number_of_models,
        figsize=(5 * number_of_models, 4.8),
        squeeze=False
    )

    axes = axes.flatten()

    display_labels = [
        "Low",
        "Medium",
        "High",
        "Very High"
    ]

    for ax, result in zip(axes, results):

        # Convert the stored Python list into a NumPy array.
        matrix = np.asarray(
            result["confusion_matrix"],
            dtype=int
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=display_labels
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(result["model"])
        ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()

    fig.savefig(
        reports_dir / "confusion_matrices.png",
        dpi=150
    )

    plt.close(fig)

    # Save model comparison metrics as CSV
    comparison_data = [
        {
            "Model": result["model"],
            "Accuracy": result["accuracy"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1-score": result["f1_score"],
        }
        for result in results
    ]

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df.to_csv(
        reports_dir / "model_comparison.csv",
        index=False
    )

    print("\nEvaluation reports saved successfully:")
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Accuracy chart: {reports_dir / 'model_comparison.png'}")
    print(f"  Confusion matrices: {reports_dir / 'confusion_matrices.png'}")
    print(f"  Comparison CSV: {reports_dir / 'model_comparison.csv'}")
