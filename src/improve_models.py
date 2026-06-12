import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import load_config
from src.data import load_train_validation_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve readmission models with threshold search and XGBoost tuning."
    )
    parser.add_argument(
        "--config",
        default="configs/xgboost_basic.yaml",
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Number of RandomizedSearchCV parameter samples for tuned XGBoost.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of stratified CV folds for tuning.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "recall", "precision"],
        default="f1",
        help="Metric used to choose the best validation threshold.",
    )
    parser.add_argument(
        "--output-prefix",
        default="improved",
        help="Prefix for report and model output files.",
    )
    return parser.parse_args()


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    if positive_count == 0:
        raise ValueError("Cannot compute scale_pos_weight because there are no positive labels.")
    return negative_count / positive_count


def evaluate_predictions(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
    }


def search_thresholds(
    y_true: pd.Series,
    y_probability: pd.Series,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for step in range(1, 100):
        threshold = step / 100
        rows.append(evaluate_predictions(y_true, y_probability, threshold))

    threshold_frame = pd.DataFrame(rows)
    best_index = threshold_frame[metric].idxmax()
    best_row = threshold_frame.loc[best_index].to_dict()
    return threshold_frame, best_row


def build_xgboost_params(config: dict, scale_pos_weight: float) -> dict[str, Any]:
    model_config = config["model"]
    params = dict(model_config["params"])
    params["random_state"] = model_config.get("random_state", 42)
    params["scale_pos_weight"] = scale_pos_weight
    return params


def build_original_xgboost_params(config: dict) -> dict[str, Any]:
    model_config = config["model"]
    params = dict(model_config["params"])
    params["random_state"] = model_config.get("random_state", 42)
    return params


def build_tuned_xgboost_search(
    config: dict,
    scale_pos_weight: float,
    cv: int,
    n_iter: int,
) -> RandomizedSearchCV:
    params = build_xgboost_params(config, scale_pos_weight)
    random_state = int(config["model"].get("random_state", 42))
    estimator = XGBClassifier(**params)

    param_distributions = {
        "n_estimators": [100, 200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.01, 0.1, 0.5],
    }
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=splitter,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def evaluate_model(
    name: str,
    model: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    probabilities = pd.Series(model.predict_proba(x_val)[:, 1], name="predicted_probability")
    threshold_frame, best_threshold = search_thresholds(y_val, probabilities, threshold_metric)
    fixed_threshold_metrics = evaluate_predictions(y_val, probabilities, 0.5)

    row = {
        "model": name,
        "roc_auc": fixed_threshold_metrics["roc_auc"],
        "average_precision": fixed_threshold_metrics["average_precision"],
        "accuracy_at_0_50": fixed_threshold_metrics["accuracy"],
        "precision_at_0_50": fixed_threshold_metrics["precision"],
        "recall_at_0_50": fixed_threshold_metrics["recall"],
        "f1_at_0_50": fixed_threshold_metrics["f1"],
        "best_threshold": best_threshold["threshold"],
        f"best_{threshold_metric}": best_threshold[threshold_metric],
        "precision_at_best_threshold": best_threshold["precision"],
        "recall_at_best_threshold": best_threshold["recall"],
        "f1_at_best_threshold": best_threshold["f1"],
        "accuracy_at_best_threshold": best_threshold["accuracy"],
        "tp_at_best_threshold": best_threshold["tp"],
        "fp_at_best_threshold": best_threshold["fp"],
        "fn_at_best_threshold": best_threshold["fn"],
        "tn_at_best_threshold": best_threshold["tn"],
    }
    threshold_frame.insert(0, "model", name)
    return row, threshold_frame, probabilities


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    y_pred = (y_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Best Model Confusion Matrix (threshold={threshold:.2f})")
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
    ax.set_title("Best Model ROC Curve")
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
    ax.set_title("Best Model Precision-Recall Curve")
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


def save_feature_importance_plot(model: Any, feature_names: list[str], output_path: Path) -> None:
    importance = get_feature_importance(model, feature_names)
    if importance is None:
        return

    importance = importance.tail(20)
    fig_height = max(4, len(importance) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    importance.plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title("Best Model Feature Importance")
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

    save_confusion_matrix_plot(y_val, best_probabilities, best_threshold, plot_paths["confusion_matrix"])
    save_roc_curve_plot(y_val, best_probabilities, plot_paths["roc_curve"])
    save_precision_recall_curve_plot(y_val, best_probabilities, plot_paths["precision_recall_curve"])
    save_feature_importance_plot(best_model, list(best_validation_features.columns), plot_paths["feature_importance"])
    save_model_comparison_plot(comparison, plot_paths["model_comparison"])
    save_threshold_metric_plot(
        all_thresholds,
        threshold_metric,
        best_model_name,
        plot_paths["threshold_search"],
    )
    save_probability_distribution_plot(y_val, best_probabilities, plot_paths["probability_distribution"])

    return {name: str(path) for name, path in plot_paths.items() if path.exists()}


def save_markdown_report(
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    plot_paths: dict[str, str],
    output_path: Path,
) -> None:
    metrics = summary["best_model_metrics"]
    table_columns = [
        "model",
        "roc_auc",
        "average_precision",
        "best_threshold",
        "precision_at_best_threshold",
        "recall_at_best_threshold",
        "f1_at_best_threshold",
    ]
    table_frame = comparison[table_columns].copy()
    for column in table_columns[1:]:
        table_frame[column] = table_frame[column].map(lambda value: f"{value:.4f}")
    markdown_table = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for _, row in table_frame.iterrows():
        markdown_table.append("| " + " | ".join(str(row[column]) for column in table_columns) + " |")

    lines = [
        "# Improved Readmission Model Report",
        "",
        "## Best Model",
        "",
        f"- Model: `{summary['best_model']}`",
        f"- Selected threshold: `{summary['best_threshold']:.2f}`",
        f"- Scale pos weight: `{summary['scale_pos_weight']:.4f}`",
        f"- ROC AUC: `{metrics['roc_auc']:.4f}`",
        f"- Average precision: `{metrics['average_precision']:.4f}`",
        f"- Precision at best threshold: `{metrics['precision_at_best_threshold']:.4f}`",
        f"- Recall at best threshold: `{metrics['recall_at_best_threshold']:.4f}`",
        f"- F1 at best threshold: `{metrics['f1_at_best_threshold']:.4f}`",
        "",
        "## Confusion Matrix At Best Threshold",
        "",
        f"- TP: `{int(metrics['tp_at_best_threshold'])}`",
        f"- FP: `{int(metrics['fp_at_best_threshold'])}`",
        f"- FN: `{int(metrics['fn_at_best_threshold'])}`",
        f"- TN: `{int(metrics['tn_at_best_threshold'])}`",
        "",
        "## Model Comparison",
        "",
        *markdown_table,
        "",
        "## Figures",
        "",
    ]

    for name, path in plot_paths.items():
        lines.append(f"- {name}: `{path}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    x_train, y_train, x_val, y_val = load_train_validation_data(config)

    output_config = config["outputs"]
    report_dir = Path(output_config["report_dir"])
    model_dir = Path(output_config["model_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    original_xgboost_params = build_original_xgboost_params(config)
    xgboost_params = build_xgboost_params(config, scale_pos_weight)

    models: list[tuple[str, Any, pd.DataFrame, pd.DataFrame]] = []

    xgb_original = XGBClassifier(**original_xgboost_params)
    xgb_original.fit(x_train, y_train)
    models.append(("xgboost_config_original", xgb_original, x_train, x_val))

    tuned_search = build_tuned_xgboost_search(config, scale_pos_weight, args.cv, args.n_iter)
    tuned_search.fit(x_train, y_train)
    tuned_model = tuned_search.best_estimator_
    models.append(
        (
            "xgboost_tuned_weighted",
            tuned_model,
            x_train,
            x_val,
        )
    )

    comparison_rows = []
    threshold_frames = []
    model_probabilities: dict[str, pd.Series] = {}
    model_validation_features: dict[str, pd.DataFrame] = {}
    model_training_features: dict[str, pd.DataFrame] = {}
    model_objects: dict[str, Any] = {}

    for name, model, train_features, validation_features in models:
        row, threshold_frame, probabilities = evaluate_model(
            name,
            model,
            validation_features,
            y_val,
            args.threshold_metric,
        )
        comparison_rows.append(row)
        threshold_frames.append(threshold_frame)
        model_probabilities[name] = probabilities
        model_validation_features[name] = validation_features
        model_training_features[name] = train_features
        model_objects[name] = model

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=[f"best_{args.threshold_metric}", "average_precision", "roc_auc"],
        ascending=False,
    )
    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    best_model_name = str(comparison.iloc[0]["model"])
    best_threshold = float(comparison.iloc[0]["best_threshold"])
    best_model = model_objects[best_model_name]
    best_probabilities = model_probabilities[best_model_name]
    best_validation_features = model_validation_features[best_model_name]
    best_training_features = model_training_features[best_model_name]

    comparison_path = report_dir / f"{args.output_prefix}_model_comparison.csv"
    thresholds_path = report_dir / f"{args.output_prefix}_threshold_search.csv"
    predictions_path = report_dir / f"{args.output_prefix}_best_val_predictions.csv"
    summary_path = report_dir / f"{args.output_prefix}_summary.json"
    report_path = report_dir / f"{args.output_prefix}_report.md"
    model_path = model_dir / f"{args.output_prefix}_best_model.joblib"
    plot_dir = Path(output_config.get("plot_dir", report_dir / "figures")) / args.output_prefix

    comparison.to_csv(comparison_path, index=False)
    all_thresholds.to_csv(thresholds_path, index=False)
    pd.DataFrame(
        {
            "actual": y_val,
            "predicted_probability": best_probabilities,
            "predicted_label": (best_probabilities >= best_threshold).astype(int),
        }
    ).to_csv(predictions_path, index=False)

    plot_paths = save_improvement_plots(
        comparison=comparison,
        all_thresholds=all_thresholds,
        best_model_name=best_model_name,
        best_model=best_model,
        best_probabilities=best_probabilities,
        best_validation_features=best_validation_features,
        y_val=y_val,
        threshold_metric=args.threshold_metric,
        best_threshold=best_threshold,
        plot_dir=plot_dir,
    )

    summary = {
        "scale_pos_weight": scale_pos_weight,
        "threshold_metric": args.threshold_metric,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "best_model_metrics": comparison.iloc[0].to_dict(),
        "tuned_xgboost_best_cv_average_precision": float(tuned_search.best_score_),
        "tuned_xgboost_best_params": tuned_search.best_params_,
        "uses_original_filtered_features_only": True,
        "improvement_method": "XGBoost tuned with scale_pos_weight using original filtered features only.",
        "outputs": {
            "comparison": str(comparison_path),
            "thresholds": str(thresholds_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "model": str(model_path),
            "plots": plot_paths,
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_markdown_report(summary, comparison, plot_paths, report_path)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": list(best_training_features.columns),
            "raw_feature_columns": list(x_train.columns),
            "selected_threshold": best_threshold,
            "config": config,
            "summary": summary,
        },
        model_path,
    )

    print(f"scale_pos_weight={scale_pos_weight:.4f}")
    print(f"Best model: {best_model_name}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Model comparison saved to: {comparison_path}")
    print(f"Threshold search saved to: {thresholds_path}")
    print(f"Best validation predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Markdown report saved to: {report_path}")
    print(f"Plots saved to: {plot_dir}")
    print(f"Best model saved to: {model_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"- {plot_name}: {plot_path}")
    print("\nTop comparison rows:")
    print(comparison.head().to_string(index=False))


if __name__ == "__main__":
    main()
