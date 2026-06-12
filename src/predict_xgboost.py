import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict readmission probability with a trained model.")
    parser.add_argument(
        "--model",
        default="models/xgboost_basic.joblib",
        help="Path to the trained joblib model bundle.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV file containing feature columns.",
    )
    parser.add_argument(
        "--output",
        default="reports/new_predictions.csv",
        help="Path where predictions will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for class label 1. Defaults to the saved model threshold, then 0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = args.threshold
    if threshold is None:
        threshold = float(bundle.get("selected_threshold", 0.5))

    data = pd.read_csv(args.input)
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    features = data[feature_columns]
    probabilities = pd.Series(model.predict_proba(features)[:, 1], name="readmission_probability")

    predictions = data.copy()
    predictions["predicted_probability"] = probabilities
    predictions["predicted_label"] = (probabilities >= threshold).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Threshold used: {threshold:.4f}")


if __name__ == "__main__":
    main()
