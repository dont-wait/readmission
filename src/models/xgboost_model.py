from xgboost import XGBClassifier


def build_xgboost_classifier(config: dict) -> XGBClassifier:
    model_config = config["model"]
    params = dict(model_config["params"])
    params["random_state"] = model_config.get("random_state", 42)

    return XGBClassifier(**params)
