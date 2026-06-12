from pathlib import Path

import pandas as pd


def load_feature_target_pair(
    x_path: str | Path,
    y_path: str | Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_path)
    y_frame = pd.read_csv(y_path)

    if target_column not in y_frame.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {y_path}. "
            f"Available columns: {list(y_frame.columns)}"
        )

    y = y_frame[target_column]

    if len(x) != len(y):
        raise ValueError(
            f"Feature/target row mismatch: {x_path} has {len(x)} rows, "
            f"{y_path} has {len(y)} rows."
        )

    return x, y


def load_train_validation_data(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data_config = config["data"]
    target_column = data_config["target_column"]

    x_train, y_train = load_feature_target_pair(
        data_config["x_train_path"],
        data_config["y_train_path"],
        target_column,
    )
    x_val, y_val = load_feature_target_pair(
        data_config["x_val_path"],
        data_config["y_val_path"],
        target_column,
    )

    if list(x_train.columns) != list(x_val.columns):
        raise ValueError("Train and validation feature columns do not match.")

    return x_train, y_train, x_val, y_val
