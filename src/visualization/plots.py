from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    y_pred = (y_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
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


def save_roc_curve_plot(
    y_true: pd.Series,
    y_probability: pd.Series,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc_score = roc_auc_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(title)
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
    title: str = "Precision-Recall Curve",
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    avg_precision = average_precision_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"Average precision = {avg_precision:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_feature_importance(model: Any, feature_names: list[str]) -> pd.Series | None:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_names).sort_values()

    if hasattr(estimator, "coef_"):
        coefficients = abs(estimator.coef_[0])
        return pd.Series(coefficients, index=feature_names).sort_values()

    return None


def save_feature_importance_plot(
    model: Any,
    feature_names: list[str],
    output_path: Path,
    title: str = "Feature Importance",
) -> None:
    importance = get_feature_importance(model, feature_names)
    if importance is None:
        return

    importance = importance.tail(20)
    fig_height = max(4, len(importance) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    importance.plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_comparison_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    plot_frame = comparison.set_index("model")[
        ["roc_auc", "average_precision", "recall_at_best_threshold", "f1_at_best_threshold"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_frame.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_threshold_metric_plot(
    all_thresholds: pd.DataFrame,
    metric: str,
    best_model_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, model_frame in all_thresholds.groupby("model"):
        alpha = 1.0 if model_name == best_model_name else 0.35
        linewidth = 2.5 if model_name == best_model_name else 1.2
        ax.plot(
            model_frame["threshold"],
            model_frame[metric],
            label=model_name,
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.set_title(f"Threshold Search ({metric})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_probability_distribution_plot(
    y_true: pd.Series,
    y_probability: pd.Series,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_probability[y_true == 0], bins=20, alpha=0.65, label="Actual 0")
    ax.hist(y_probability[y_true == 1], bins=20, alpha=0.65, label="Actual 1")
    ax.set_title("Best Model Probability Distribution")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_metric_value(metric: str, value: float) -> str:
    if metric in {"False negatives", "False positives"}:
        return str(int(round(value)))
    if metric == "Best threshold":
        return f"{value:.2f}"
    return f"{value:.4f}"


def save_key_metrics_plot(key_metrics: pd.DataFrame, output_path: Path) -> None:
    if key_metrics.empty:
        return

    display_columns = ["metric", "original_xgboost", "improved_xgboost", "change", "how_to_read"]
    display_frame = key_metrics[display_columns].copy().astype("object")
    for index, row in display_frame.iterrows():
        metric = str(row["metric"])
        display_frame.at[index, "original_xgboost"] = format_metric_value(
            metric, float(row["original_xgboost"])
        )
        display_frame.at[index, "improved_xgboost"] = format_metric_value(
            metric, float(row["improved_xgboost"])
        )
        display_frame.at[index, "change"] = f"{float(row['change']):+.4f}"

    display_frame = display_frame.rename(
        columns={
            "metric": "Metric",
            "original_xgboost": "Original",
            "improved_xgboost": "Improved",
            "change": "Change",
            "how_to_read": "How to read",
        }
    )

    fig_height = max(4.5, len(display_frame) * 0.65)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="left",
        colWidths=[0.16, 0.09, 0.09, 0.08, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#345995")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F6F8")

    ax.set_title("Key Model Metrics Report", fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_basic_evaluation_plots(
    model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    y_probability: pd.Series,
    threshold: float,
    plot_dir: Path,
    feature_importance_title: str = "Feature Importance",
) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "confusion_matrix": plot_dir / "confusion_matrix.png",
        "roc_curve": plot_dir / "roc_curve.png",
        "precision_recall_curve": plot_dir / "precision_recall_curve.png",
        "feature_importance": plot_dir / "feature_importance.png",
    }

    save_confusion_matrix_plot(y_val, y_probability, threshold, plot_paths["confusion_matrix"])
    save_roc_curve_plot(y_val, y_probability, plot_paths["roc_curve"])
    save_precision_recall_curve_plot(y_val, y_probability, plot_paths["precision_recall_curve"])
    save_feature_importance_plot(
        model,
        list(x_val.columns),
        plot_paths["feature_importance"],
        title=feature_importance_title,
    )

    return {name: str(path) for name, path in plot_paths.items()}


def save_improvement_plots(
    comparison: pd.DataFrame,
    all_thresholds: pd.DataFrame,
    best_model_name: str,
    best_model: Any,
    best_probabilities: pd.Series,
    best_validation_features: pd.DataFrame,
    y_val: pd.Series,
    threshold_metric: str,
    best_threshold: float,
    plot_dir: Path,
) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "confusion_matrix": plot_dir / "confusion_matrix.png",
        "roc_curve": plot_dir / "roc_curve.png",
        "precision_recall_curve": plot_dir / "precision_recall_curve.png",
        "feature_importance": plot_dir / "feature_importance.png",
        "model_comparison": plot_dir / "model_comparison.png",
        "threshold_search": plot_dir / "threshold_search.png",
        "probability_distribution": plot_dir / "probability_distribution.png",
    }

    save_confusion_matrix_plot(
        y_val,
        best_probabilities,
        best_threshold,
        plot_paths["confusion_matrix"],
        title=f"Best Model Confusion Matrix (threshold={best_threshold:.2f})",
    )
    save_roc_curve_plot(
        y_val,
        best_probabilities,
        plot_paths["roc_curve"],
        title="Best Model ROC Curve",
    )
    save_precision_recall_curve_plot(
        y_val,
        best_probabilities,
        plot_paths["precision_recall_curve"],
        title="Best Model Precision-Recall Curve",
    )
    save_feature_importance_plot(
        best_model,
        list(best_validation_features.columns),
        plot_paths["feature_importance"],
        title="Best Model Feature Importance",
    )
    save_model_comparison_plot(comparison, plot_paths["model_comparison"])
    save_threshold_metric_plot(
        all_thresholds,
        threshold_metric,
        best_model_name,
        plot_paths["threshold_search"],
    )
    save_probability_distribution_plot(y_val, best_probabilities, plot_paths["probability_distribution"])

    return {name: str(path) for name, path in plot_paths.items() if path.exists()}
