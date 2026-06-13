import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.data import load_raw_preprocessed_data, load_test_data, load_train_validation_data
from src.visualization.plots import (
    save_basic_evaluation_plots,
    save_model_comparison_plot,
    save_probability_distribution_plot,
    save_threshold_metric_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train regularized Logistic Regression baselines for readmission prediction."
    )
    parser.add_argument(
        "--config",
        default="configs/xgboost_basic.yaml",
        help="Path to the training config YAML file.",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "recall", "precision"],
        default="f1",
        help="Metric used to choose the best validation threshold.",
    )
    return parser.parse_args()


def evaluate_predictions(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
    }


def search_thresholds(
    model_name: str,
    y_true: pd.Series,
    y_probability: pd.Series,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for step in range(1, 100):
        threshold = step / 100
        row = evaluate_predictions(y_true, y_probability, threshold)
        row["model"] = model_name
        rows.append(row)

    threshold_frame = pd.DataFrame(rows)
    best_row = threshold_frame.loc[threshold_frame[metric].idxmax()].to_dict()
    return threshold_frame, best_row


def build_logistic_model(
    name: str,
    class_weight: str | None,
    cv: int,
    random_state: int,
) -> LogisticRegressionCV:
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        cv=splitter,
        scoring="average_precision",
        l1_ratios=[0.0],
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=10000,
        n_jobs=-1,
        refit=True,
        use_legacy_attributes=True,
    )


def make_comparison_row(
    model_name: str,
    probabilities: pd.Series,
    y_val: pd.Series,
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    threshold_frame, best_threshold = search_thresholds(
        model_name,
        y_val,
        probabilities,
        threshold_metric,
    )
    fixed_threshold_metrics = evaluate_predictions(y_val, probabilities, 0.5)

    row = {
        "model": model_name,
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
    return row, threshold_frame


def build_coefficients_frame(
    model: LogisticRegressionCV,
    feature_columns: list[str],
) -> pd.DataFrame:
    coefficients = pd.Series(model.coef_[0], index=feature_columns, name="coefficient")
    frame = coefficients.reset_index().rename(columns={"index": "feature"})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame["effect"] = frame["coefficient"].map(
        lambda value: "increases_risk" if value > 0 else "decreases_risk"
    )
    return frame.sort_values("abs_coefficient", ascending=False)


def save_markdown_report(
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = summary["best_model_metrics"]
    test_metrics = summary.get("test_metrics")
    table_columns = [
        "model",
        "roc_auc",
        "average_precision",
        "best_threshold",
        "precision_at_best_threshold",
        "recall_at_best_threshold",
        "f1_at_best_threshold",
    ]
    comparison_display = comparison[table_columns].copy()
    for column in table_columns[1:]:
        comparison_display[column] = comparison_display[column].map(lambda value: f"{value:.4f}")

    top_positive = coefficients.sort_values("coefficient", ascending=False).head(6)
    top_negative = coefficients.sort_values("coefficient", ascending=True).head(6)

    lines = [
        "# Logistic Regression Report",
        "",
        "## Best Model",
        "",
        f"- Model: `{summary['best_model']}`",
        f"- Selected threshold: `{summary['best_threshold']:.2f}`",
        f"- Selected C: `{summary['best_c']:.4g}`",
        f"- ROC AUC: `{metrics['roc_auc']:.4f}`",
        f"- Average precision: `{metrics['average_precision']:.4f}`",
        f"- Precision at best threshold: `{metrics['precision_at_best_threshold']:.4f}`",
        f"- Recall at best threshold: `{metrics['recall_at_best_threshold']:.4f}`",
        f"- F1 at best threshold: `{metrics['f1_at_best_threshold']:.4f}`",
        "",
        "## Test Set Metrics",
        "",
        *(
            [
                f"- Threshold: `{test_metrics['threshold']:.2f}`",
                f"- Accuracy: `{test_metrics['accuracy']:.4f}`",
                f"- Precision: `{test_metrics['precision']:.4f}`",
                f"- Recall: `{test_metrics['recall']:.4f}`",
                f"- F1: `{test_metrics['f1']:.4f}`",
                f"- ROC AUC: `{test_metrics['roc_auc']:.4f}`",
                f"- Average precision: `{test_metrics['average_precision']:.4f}`",
                f"- Confusion matrix: TP `{test_metrics['tp']}`, FP `{test_metrics['fp']}`, "
                f"FN `{test_metrics['fn']}`, TN `{test_metrics['tn']}`",
            ]
            if test_metrics
            else ["- Test set is not configured."]
        ),
        "",
        "## Model Comparison",
        "",
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
        *[
            "| " + " | ".join(str(row[column]) for column in table_columns) + " |"
            for _, row in comparison_display.iterrows()
        ],
        "",
        "## Coefficients",
        "",
        "Positive coefficients increase predicted readmission risk; negative coefficients decrease it.",
        "",
        "### Largest Positive Coefficients",
        "",
        "| feature | coefficient |",
        "| --- | --- |",
        *[
            f"| {row['feature']} | {float(row['coefficient']):.4f} |"
            for _, row in top_positive.iterrows()
        ],
        "",
        "### Largest Negative Coefficients",
        "",
        "| feature | coefficient |",
        "| --- | --- |",
        *[
            f"| {row['feature']} | {float(row['coefficient']):.4f} |"
            for _, row in top_negative.iterrows()
        ],
        "",
        "## Outputs",
        "",
    ]

    output_keys = [
        "comparison",
        "thresholds",
        "validation_predictions",
        "test_metrics",
        "test_predictions",
        "coefficients",
        "summary",
        "report",
        "model",
    ]
    for name in output_keys:
        path = summary["outputs"].get(name)
        if path:
            lines.append(f"- {name}: `{path}`")

    lines.extend(["", "## Figures", ""])
    for name, path in summary["outputs"]["plots"].items():
        lines.append(f"- {name}: `{path}`")

    if summary["outputs"]["test_plots"]:
        lines.extend(["", "## Test Figures", ""])
        for name, path in summary["outputs"]["test_plots"].items():
            lines.append(f"- {name}: `{path}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_data = load_raw_preprocessed_data(config)
    if raw_data is not None:
        x_train = raw_data["x_train"]
        y_train = raw_data["y_train"]
        x_val = raw_data["x_val"]
        y_val = raw_data["y_val"]
        preprocessing = raw_data["preprocessing"]
    else:
        x_train, y_train, x_val, y_val = load_train_validation_data(config)
        preprocessing = None
    output_config = config["outputs"]
    random_state = int(config["model"].get("random_state", 42))

    report_dir = Path(output_config.get("logistic_report_dir", "reports/logistic"))
    plot_dir = Path(output_config.get("logistic_plot_dir", report_dir / "figures"))
    model_dir = Path(output_config["model_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    candidate_specs = [
        ("logistic_l2", None),
        ("logistic_l2_balanced", "balanced"),
    ]
    comparison_rows = []
    threshold_frames = []
    probabilities_by_model: dict[str, pd.Series] = {}
    models: dict[str, LogisticRegressionCV] = {}

    for model_name, class_weight in candidate_specs:
        model = build_logistic_model(model_name, class_weight, args.cv, random_state)
        model.fit(x_train, y_train)
        probabilities = pd.Series(
            model.predict_proba(x_val)[:, 1],
            name="predicted_probability",
        )
        row, threshold_frame = make_comparison_row(
            model_name,
            probabilities,
            y_val,
            args.threshold_metric,
        )
        row["selected_c"] = float(model.C_[0])
        row["class_weight"] = class_weight or "none"
        comparison_rows.append(row)
        threshold_frames.append(threshold_frame)
        probabilities_by_model[model_name] = probabilities
        models[model_name] = model

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=[f"best_{args.threshold_metric}", "average_precision", "roc_auc"],
        ascending=False,
    )
    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    best_model_name = str(comparison.iloc[0]["model"])
    best_model = models[best_model_name]
    best_threshold = float(comparison.iloc[0]["best_threshold"])
    best_probabilities = probabilities_by_model[best_model_name]
    coefficients = build_coefficients_frame(best_model, list(x_train.columns))

    comparison_path = report_dir / "model_comparison.csv"
    thresholds_path = report_dir / "threshold_search.csv"
    predictions_path = report_dir / "best_val_predictions.csv"
    coefficients_path = report_dir / "coefficients.csv"
    summary_path = report_dir / "summary.json"
    report_path = report_dir / "report.md"
    model_path = model_dir / "logistic_regression_best_model.joblib"

    comparison.to_csv(comparison_path, index=False)
    all_thresholds.to_csv(thresholds_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    pd.DataFrame(
        {
            "actual": y_val,
            "predicted_probability": best_probabilities,
            "predicted_label": (best_probabilities >= best_threshold).astype(int),
        }
    ).to_csv(predictions_path, index=False)

    plot_paths = save_basic_evaluation_plots(
        best_model,
        x_val,
        y_val,
        best_probabilities,
        best_threshold,
        plot_dir,
        feature_importance_title="Logistic Regression Coefficient Magnitudes",
    )
    comparison_plot_path = plot_dir / "model_comparison.png"
    threshold_plot_path = plot_dir / "threshold_search.png"
    probability_plot_path = plot_dir / "probability_distribution.png"
    save_model_comparison_plot(comparison, comparison_plot_path)
    save_threshold_metric_plot(all_thresholds, args.threshold_metric, best_model_name, threshold_plot_path)
    save_probability_distribution_plot(y_val, best_probabilities, probability_plot_path)
    plot_paths.update(
        {
            "model_comparison": str(comparison_plot_path),
            "threshold_search": str(threshold_plot_path),
            "probability_distribution": str(probability_plot_path),
        }
    )

    test_metrics = None
    test_predictions_path = None
    test_plot_paths: dict[str, str] = {}
    test_data = (
        (raw_data["x_test"], raw_data["y_test"])
        if raw_data is not None
        else load_test_data(config, list(x_train.columns))
    )
    if test_data is not None:
        x_test, y_test = test_data
        test_probabilities = pd.Series(
            best_model.predict_proba(x_test)[:, 1],
            name="predicted_probability",
        )
        test_metrics = evaluate_predictions(y_test, test_probabilities, best_threshold)
        test_predictions_path = report_dir / "best_test_predictions.csv"
        pd.DataFrame(
            {
                "actual": y_test,
                "predicted_probability": test_probabilities,
                "predicted_label": (test_probabilities >= best_threshold).astype(int),
            }
        ).to_csv(test_predictions_path, index=False)
        with (report_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)
        test_plot_paths = save_basic_evaluation_plots(
            best_model,
            x_test,
            y_test,
            test_probabilities,
            best_threshold,
            plot_dir / "test",
            feature_importance_title="Logistic Regression Coefficient Magnitudes",
        )

    summary = {
        "threshold_metric": args.threshold_metric,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "best_c": float(best_model.C_[0]),
        "best_model_metrics": comparison.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "candidate_models": [name for name, _ in candidate_specs],
        "outputs": {
            "comparison": str(comparison_path),
            "thresholds": str(thresholds_path),
            "validation_predictions": str(predictions_path),
            "test_metrics": str(report_dir / "test_metrics.json") if test_metrics else None,
            "test_predictions": str(test_predictions_path) if test_predictions_path else None,
            "coefficients": str(coefficients_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "model": str(model_path),
            "plots": plot_paths,
            "test_plots": test_plot_paths,
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    save_markdown_report(summary, comparison, coefficients, report_path)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": list(x_train.columns),
            "raw_feature_columns": list(x_train.columns),
            "preprocessing": preprocessing,
            "selected_threshold": best_threshold,
            "config": config,
            "summary": summary,
        },
        model_path,
    )

    print(f"Best model: {best_model_name}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Selected C: {float(best_model.C_[0]):.4g}")
    print(f"Model comparison saved to: {comparison_path}")
    print(f"Coefficients saved to: {coefficients_path}")
    print(f"Markdown report saved to: {report_path}")
    print(f"Best model saved to: {model_path}")
    if test_metrics:
        print(
            "Test metrics: "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"roc_auc={test_metrics['roc_auc']:.4f}"
        )
    print("\nTop comparison rows:")
    print(comparison.head().to_string(index=False))


if __name__ == "__main__":
    main()
