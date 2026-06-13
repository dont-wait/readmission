import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import load_config
from src.data import load_raw_preprocessed_data, load_test_data, load_train_validation_data
from src.visualization.plots import (
    format_metric_value,
    save_basic_evaluation_plots,
    save_improvement_plots,
    save_key_metrics_plot,
)


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


def build_key_metrics_table(comparison: pd.DataFrame, best_model_name: str) -> pd.DataFrame:
    baseline_row = comparison[comparison["model"] == "xgboost_config_original"]
    best_row = comparison[comparison["model"] == best_model_name]

    if baseline_row.empty or best_row.empty:
        return pd.DataFrame()

    baseline = baseline_row.iloc[0]
    best = best_row.iloc[0]
    metric_specs = [
        (
            "ROC AUC",
            "roc_auc",
            "Cang cao cang tot. 0.5 gan nhu doan ngau nhien; tren 0.7 la kha hon.",
            "Kha nang xep hang ca nguy co cao hon ca nguy co thap.",
        ),
        (
            "Average Precision",
            "average_precision",
            "Cang cao cang tot. Nen doc kem precision-recall curve khi lop 1 quan trong.",
            "Chat luong xep hang rieng cho lop tai nhap vien.",
        ),
        (
            "F1 at best threshold",
            "f1_at_best_threshold",
            "Cang cao cang tot. Can bang giua precision va recall.",
            "Chi so tong hop tai threshold da chon.",
        ),
        (
            "Recall at best threshold",
            "recall_at_best_threshold",
            "Cang cao cang it bo sot ca tai nhap vien.",
            "Ty le ca tai nhap vien that duoc phat hien.",
        ),
        (
            "Precision at best threshold",
            "precision_at_best_threshold",
            "Cang cao cang it canh bao nham.",
            "Trong cac ca du doan tai nhap vien, bao nhieu ca dung.",
        ),
        (
            "Best threshold",
            "best_threshold",
            "Nguong cat xac suat thanh nhan 0/1. Khong phai tham so train.",
            "Nguong dung de tao predicted_label.",
        ),
        (
            "False negatives",
            "fn_at_best_threshold",
            "Cang thap cang tot, dac biet voi bai toan y te.",
            "So ca tai nhap vien bi model bo sot.",
        ),
        (
            "False positives",
            "fp_at_best_threshold",
            "Cang thap cang tot, nhung thuong tang khi muon recall cao.",
            "So ca khong tai nhap vien bi canh bao nham.",
        ),
    ]

    rows = []
    for metric_name, column, how_to_read, meaning in metric_specs:
        baseline_value = float(baseline[column])
        best_value = float(best[column])
        rows.append(
            {
                "metric": metric_name,
                "original_xgboost": baseline_value,
                "improved_xgboost": best_value,
                "change": best_value - baseline_value,
                "meaning": meaning,
                "how_to_read": how_to_read,
            }
        )

    return pd.DataFrame(rows)


def save_markdown_report(
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    key_metrics: pd.DataFrame,
    plot_paths: dict[str, str],
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
        "## How To Read The Main Metrics",
        "",
        "- `ROC AUC`: kha nang xep hang mau duong cao hon mau am. Cang cao cang tot.",
        "- `Average Precision`: nen doc khi lop tai nhap vien quan trong. Cang cao cang tot.",
        "- `Recall`: kha nang bat duoc ca tai nhap vien that. Recall cao thi it bo sot.",
        "- `Precision`: khi model canh bao tai nhap vien, ty le canh bao dung la bao nhieu.",
        "- `F1`: can bang giua precision va recall tai threshold da chon.",
        "- `Best threshold`: nguong cat xac suat thanh nhan 0/1, khong phai tham so train.",
        "",
        "## Key Metrics Table",
        "",
        "| metric | original_xgboost | improved_xgboost | change | how_to_read |",
        "| --- | --- | --- | --- | --- |",
        *[
            "| "
            + " | ".join(
                [
                    str(row["metric"]),
                    format_metric_value(str(row["metric"]), float(row["original_xgboost"])),
                    format_metric_value(str(row["metric"]), float(row["improved_xgboost"])),
                    f"{float(row['change']):+.4f}",
                    str(row["how_to_read"]),
                ]
            )
            + " |"
            for _, row in key_metrics.iterrows()
        ],
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


def build_output_filename(prefix: str, filename: str) -> str:
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


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
    report_dir = Path(output_config.get("improved_report_dir", output_config["report_dir"]))
    model_dir = Path(output_config["model_dir"])
    output_prefix = str(output_config.get("improved_output_prefix", args.output_prefix))
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

    comparison_path = report_dir / build_output_filename(output_prefix, "model_comparison.csv")
    thresholds_path = report_dir / build_output_filename(output_prefix, "threshold_search.csv")
    predictions_path = report_dir / build_output_filename(output_prefix, "best_val_predictions.csv")
    key_metrics_path = report_dir / build_output_filename(output_prefix, "key_metrics.csv")
    summary_path = report_dir / build_output_filename(output_prefix, "summary.json")
    report_path = report_dir / build_output_filename(output_prefix, "report.md")
    model_path = model_dir / build_output_filename(args.output_prefix, "best_model.joblib")
    plot_dir = Path(output_config.get("improved_plot_dir", report_dir / "figures"))
    key_metrics_plot_path = plot_dir / "key_metrics_table.png"

    comparison.to_csv(comparison_path, index=False)
    all_thresholds.to_csv(thresholds_path, index=False)
    key_metrics = build_key_metrics_table(comparison, best_model_name)
    key_metrics.to_csv(key_metrics_path, index=False)
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
    save_key_metrics_plot(key_metrics, key_metrics_plot_path)
    if key_metrics_plot_path.exists():
        plot_paths["key_metrics_table"] = str(key_metrics_plot_path)

    test_metrics = None
    test_predictions_path = None
    test_metrics_path = None
    test_plot_paths: dict[str, str] = {}
    test_data = (
        (raw_data["x_test"], raw_data["y_test"])
        if raw_data is not None
        else load_test_data(config, list(best_training_features.columns))
    )
    if test_data is not None:
        x_test, y_test = test_data
        test_probabilities = pd.Series(
            best_model.predict_proba(x_test)[:, 1],
            name="predicted_probability",
        )
        test_metrics = evaluate_predictions(y_test, test_probabilities, best_threshold)
        test_predictions_path = report_dir / build_output_filename(
            output_prefix,
            "best_test_predictions.csv",
        )
        test_metrics_path = report_dir / build_output_filename(output_prefix, "test_metrics.json")
        pd.DataFrame(
            {
                "actual": y_test,
                "predicted_probability": test_probabilities,
                "predicted_label": (test_probabilities >= best_threshold).astype(int),
            }
        ).to_csv(test_predictions_path, index=False)
        with test_metrics_path.open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)
        test_plot_paths = save_basic_evaluation_plots(
            best_model,
            x_test,
            y_test,
            test_probabilities,
            best_threshold,
            plot_dir / "test",
        )

    summary = {
        "scale_pos_weight": scale_pos_weight,
        "threshold_metric": args.threshold_metric,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "best_model_metrics": comparison.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "tuned_xgboost_best_cv_average_precision": float(tuned_search.best_score_),
        "tuned_xgboost_best_params": tuned_search.best_params_,
        "uses_original_filtered_features_only": True,
        "improvement_method": (
            "Compare the original XGBoost config with tuned XGBoost using scale_pos_weight, "
            "using original filtered features only."
        ),
        "outputs": {
            "comparison": str(comparison_path),
            "key_metrics": str(key_metrics_path),
            "thresholds": str(thresholds_path),
            "predictions": str(predictions_path),
            "test_predictions": str(test_predictions_path) if test_predictions_path else None,
            "test_metrics": str(test_metrics_path) if test_metrics_path else None,
            "summary": str(summary_path),
            "report": str(report_path),
            "model": str(model_path),
            "plots": plot_paths,
            "test_plots": test_plot_paths,
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_markdown_report(summary, comparison, key_metrics, plot_paths, report_path)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": list(best_training_features.columns),
            "raw_feature_columns": list(x_train.columns),
            "preprocessing": preprocessing,
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
    print(f"Key metrics report saved to: {key_metrics_path}")
    print(f"Threshold search saved to: {thresholds_path}")
    print(f"Best validation predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Markdown report saved to: {report_path}")
    print(f"Plots saved to: {plot_dir}")
    if test_metrics_path and test_predictions_path:
        print(f"Test metrics saved to: {test_metrics_path}")
        print(f"Best test predictions saved to: {test_predictions_path}")
        print(f"Test plots saved to: {plot_dir / 'test'}")
    print(f"Best model saved to: {model_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"- {plot_name}: {plot_path}")
    print("\nTop comparison rows:")
    print(comparison.head().to_string(index=False))


if __name__ == "__main__":
    main()
