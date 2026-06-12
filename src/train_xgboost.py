import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

from src.config import load_config
from src.data import load_train_validation_data
from src.models.xgboost_model import build_xgboost_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic XGBoost readmission model.")
    parser.add_argument(
        "--config",
        default="configs/xgboost_basic.yaml",
        help="Path to the training config YAML file.",
    )
    return parser.parse_args()


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float,
) -> dict:
    y_pred = (y_probability >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["No", "Yes"])
    ax.set_yticks([0, 1], labels=["No", "Yes"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true: pd.Series, y_probability: pd.Series, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc_score = roc_auc_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_precision_recall_curve_plot(
    y_true: pd.Series,
    y_probability: pd.Series,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    avg_precision = average_precision_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"Average precision = {avg_precision:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(model, feature_names: list[str], output_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    fig_height = max(4, len(importance) * 0.35)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    importance.plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_plots(
    model,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    y_probability: pd.Series,
    threshold: float,
    plot_dir: Path,
) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_probability >= threshold).astype(int)

    plot_paths = {
        "confusion_matrix": plot_dir / "confusion_matrix.png",
        "roc_curve": plot_dir / "roc_curve.png",
        "precision_recall_curve": plot_dir / "precision_recall_curve.png",
        "feature_importance": plot_dir / "feature_importance.png",
    }

    save_confusion_matrix_plot(y_val, y_pred, plot_paths["confusion_matrix"])
    save_roc_curve_plot(y_val, y_probability, plot_paths["roc_curve"])
    save_precision_recall_curve_plot(y_val, y_probability, plot_paths["precision_recall_curve"])
    save_feature_importance_plot(model, list(x_val.columns), plot_paths["feature_importance"])

    return {name: str(path) for name, path in plot_paths.items()}


def main() -> None:
    config = load_config(parse_args().config)
    x_train, y_train, x_val, y_val = load_train_validation_data(config)

    model = build_xgboost_classifier(config)
    training_config = config["training"]

    early_stopping_rounds = training_config.get("early_stopping_rounds")
    if early_stopping_rounds:
        model.set_params(early_stopping_rounds=early_stopping_rounds)

    model.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_val, y_val)],
        verbose=training_config.get("verbose", False),
    )

    val_probability = pd.Series(model.predict_proba(x_val)[:, 1], name="readmission_probability")
    threshold = float(training_config.get("threshold", 0.5))
    metrics = evaluate_binary_classifier(y_val, val_probability, threshold)

    output_config = config["outputs"]
    model_dir = Path(output_config["model_dir"])
    report_dir = Path(output_config["report_dir"])
    plot_dir = Path(output_config.get("plot_dir", report_dir / "figures"))
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / output_config["model_filename"]
    metrics_path = report_dir / output_config["metrics_filename"]
    predictions_path = report_dir / output_config["predictions_filename"]

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(x_train.columns),
            "config": config,
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions = pd.DataFrame(
        {
            "actual": y_val,
            "predicted_probability": val_probability,
            "predicted_label": (val_probability >= threshold).astype(int),
        }
    )
    predictions.to_csv(predictions_path, index=False)
    plot_paths = save_evaluation_plots(model, x_val, y_val, val_probability, threshold, plot_dir)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Plots saved to: {plot_dir}")
    for plot_name, plot_path in plot_paths.items():
        print(f"- {plot_name}: {plot_path}")
    print(
        "Validation metrics: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
