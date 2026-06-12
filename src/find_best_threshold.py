import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best classification threshold from predictions.")
    parser.add_argument(
        "--predictions",
        default="reports/xgboost_basic_val_predictions.csv",
        help="CSV file containing actual labels and predicted probabilities.",
    )
    parser.add_argument("--actual-column", default="actual", help="Column containing true labels.")
    parser.add_argument(
        "--probability-column",
        default="predicted_probability",
        help="Column containing predicted probability for class 1.",
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "recall", "precision"],
        default="f1",
        help="Metric used to choose the best threshold.",
    )
    parser.add_argument(
        "--output",
        default="reports/best_threshold_search.csv",
        help="Path to save all threshold metrics.",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/best_threshold_summary.json",
        help="Path to save the best threshold summary.",
    )
    return parser.parse_args()


def evaluate_threshold(
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
    }


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)

    y_true = predictions[args.actual_column]
    y_probability = predictions[args.probability_column]

    rows = [evaluate_threshold(y_true, y_probability, step / 100) for step in range(1, 100)]
    threshold_metrics = pd.DataFrame(rows)
    best_threshold = threshold_metrics.loc[threshold_metrics[args.metric].idxmax()].to_dict()

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_metrics.to_csv(output_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metric": args.metric,
                "predictions": args.predictions,
                "best_threshold": best_threshold,
                "output": str(output_path),
            },
            file,
            indent=2,
        )

    print(f"Threshold search saved to: {output_path}")
    print(f"Best threshold summary saved to: {summary_path}")
    print(
        "Best threshold: "
        f"{best_threshold['threshold']:.2f}, "
        f"precision={best_threshold['precision']:.4f}, "
        f"recall={best_threshold['recall']:.4f}, "
        f"f1={best_threshold['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
