from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "age",
    "bmi",
    "bnp",
    "sodium",
    "creatinine",
    "systolic_bp",
    "heart_rate",
    "ace_inhibitor",
    "beta_blocker",
    "diuretic",
    "adherence_score",
    "distance_to_hospital_km",
]


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


def load_test_data(
    config: dict,
    expected_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series] | None:
    data_config = config["data"]
    x_test_path = data_config.get("x_test_path")
    y_test_path = data_config.get("y_test_path")

    if not x_test_path or not y_test_path:
        return None

    x_test, y_test = load_feature_target_pair(
        x_test_path,
        y_test_path,
        data_config["target_column"],
    )

    if list(x_test.columns) != expected_columns:
        raise ValueError("Train and test feature columns do not match.")

    return x_test, y_test


def load_raw_preprocessed_data(config: dict) -> dict[str, Any] | None:
    data_config = config["data"]
    raw_data_path = data_config.get("raw_data_path")
    if not raw_data_path:
        return None

    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        return None

    target_column = data_config["target_column"]
    feature_columns = data_config.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    raw = pd.read_csv(raw_path)
    if target_column not in raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in {raw_path}.")

    if "creatinine" in raw.columns:
        raw.loc[raw["creatinine"] < 0, "creatinine"] = pd.NA

    missing_features = [column for column in feature_columns if column not in raw.columns]
    if missing_features:
        raise ValueError(f"Raw data is missing required feature columns: {missing_features}")

    x = raw.drop(columns=[target_column])
    y = raw[target_column]
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(data_config.get("split_random_state", 42)),
        stratify=y,
    )

    numeric_columns = x_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "patient_id" in numeric_columns:
        numeric_columns.remove("patient_id")

    imputer = SimpleImputer(strategy=data_config.get("imputer_strategy", "median"))
    x_train_imputed = x_train_raw.copy()
    x_test_imputed = x_test_raw.copy()
    x_train_imputed[numeric_columns] = imputer.fit_transform(x_train_imputed[numeric_columns])
    x_test_imputed[numeric_columns] = imputer.transform(x_test_imputed[numeric_columns])

    x_train_selected = x_train_imputed[feature_columns]
    x_test_selected = x_test_imputed[feature_columns]
    x_train_final_raw, x_val_raw, y_train_final, y_val = train_test_split(
        x_train_selected,
        y_train,
        test_size=float(data_config.get("validation_size", 0.2)),
        random_state=int(data_config.get("split_random_state", 42)),
        stratify=y_train,
    )

    scaler = StandardScaler()
    x_train_final = pd.DataFrame(
        scaler.fit_transform(x_train_final_raw),
        columns=feature_columns,
        index=x_train_final_raw.index,
    ).reset_index(drop=True)
    x_val = pd.DataFrame(
        scaler.transform(x_val_raw),
        columns=feature_columns,
        index=x_val_raw.index,
    ).reset_index(drop=True)
    x_test = pd.DataFrame(
        scaler.transform(x_test_selected),
        columns=feature_columns,
        index=x_test_selected.index,
    ).reset_index(drop=True)

    return {
        "x_train": x_train_final,
        "y_train": y_train_final.reset_index(drop=True),
        "x_val": x_val,
        "y_val": y_val.reset_index(drop=True),
        "x_test": x_test,
        "y_test": y_test.reset_index(drop=True),
        "preprocessing": {
            "input_type": "raw",
            "raw_feature_columns": list(feature_columns),
            "feature_columns": list(feature_columns),
            "numeric_columns": numeric_columns,
            "imputer": imputer,
            "scaler": scaler,
        },
    }
