import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import load_config
from src.data import load_test_data, load_train_validation_data
from src.models.xgboost_model import build_xgboost_classifier
from src.visualization.plots import save_basic_evaluation_plots


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
    plot_paths = save_basic_evaluation_plots(model, x_val, y_val, val_probability, threshold, plot_dir)

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

    test_data = load_test_data(config, list(x_train.columns))
    if test_data is not None:
        x_test, y_test = test_data
        test_probability = pd.Series(
            model.predict_proba(x_test)[:, 1],
            name="readmission_probability",
        )
        test_metrics = evaluate_binary_classifier(y_test, test_probability, threshold)
        test_metrics_path = report_dir / output_config.get(
            "test_metrics_filename",
            "xgboost_basic_test_metrics.json",
        )
        test_predictions_path = report_dir / output_config.get(
            "test_predictions_filename",
            "xgboost_basic_test_predictions.csv",
        )

        with test_metrics_path.open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)

        pd.DataFrame(
            {
                "actual": y_test,
                "predicted_probability": test_probability,
                "predicted_label": (test_probability >= threshold).astype(int),
            }
        ).to_csv(test_predictions_path, index=False)

        test_plot_paths = save_basic_evaluation_plots(
            model,
            x_test,
            y_test,
            test_probability,
            threshold,
            plot_dir / "test",
        )

        print(f"Test metrics saved to: {test_metrics_path}")
        print(f"Test predictions saved to: {test_predictions_path}")
        print(f"Test plots saved to: {plot_dir / 'test'}")
        for plot_name, plot_path in test_plot_paths.items():
            print(f"- test_{plot_name}: {plot_path}")
        print(
            "Test metrics: "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"roc_auc={test_metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
