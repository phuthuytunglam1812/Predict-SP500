from __future__ import annotations

import os
import pickle
import sys
import warnings
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "cpi_spx.csv")

MODEL_PATHS = {
    "surprise_model_json": os.path.join(MODEL_DIR, "model1_surprise_regression.json"),
    "direction_model_json": os.path.join(MODEL_DIR, "model2_direction_classifier.json"),
    "spx_model_joblib": os.path.join(MODEL_DIR, "model3_spx_classifier.joblib"),
    "surprise_model_pkl": os.path.join(MODEL_DIR, "model1_surprise_regression.pkl"),
    "direction_model_pkl": os.path.join(MODEL_DIR, "model2_direction_classifier.pkl"),
    "spx_model_pkl": os.path.join(MODEL_DIR, "model3_spx_classifier.pkl"),
}

FEATURE_ORDER = [
    "CPI_Surprise_Pct",
    "VIX_Return_Pct_Lag1",
    "SP500_Return_Lag1",
    "FedFunds_Rate",
    "FedFunds_Change_3M",
    "Inflation_Regime_Encoded",
]

LEGACY_FEATURE_ORDER = [
    "CPI_Surprise_Pct",
    "VIX_Return_Pct",
    "SP500_Return_Lag1",
    "FedFunds_Rate",
    "FedFunds_Change_3M",
    "Inflation_Regime_Encoded",
]

MODEL_1_2_FEATURES = [feature for feature in FEATURE_ORDER if feature != "CPI_Surprise_Pct"]
SURPRISE_CLASS_LABELS = {0: "Below", 1: "Match", 2: "Above"}
SPX_CLASS_LABELS = {0: "Down", 1: "Up"}


def directional_mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


setattr(sys.modules["__main__"], "directional_mse_loss", directional_mse_loss)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Release_Date"] = pd.to_datetime(df["Release_Date"])
    return prepare_history(df)


def encode_regime(value: Any) -> int:
    text = str(value).lower()
    if text == "high":
        return 2
    if text == "medium":
        return 1
    return 0


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    history = df.copy()
    history["Release_Date"] = pd.to_datetime(history["Release_Date"])
    history = history.sort_values("Release_Date").reset_index(drop=True)

    if "Inflation_Regime_Encoded" not in history.columns:
        history["Inflation_Regime_Encoded"] = np.select(
            [
                history["Inflation_Regime"].astype(str).str.lower().eq("high"),
                history["Inflation_Regime"].astype(str).str.lower().eq("medium"),
            ],
            [2, 1],
            default=0,
        )

    if "VIX_Return_Pct_Lag1" not in history.columns:
        history["VIX_Return_Pct_Lag1"] = history["VIX_Return_Pct"].shift(1)
    history["VIX_Return_Pct_Lag1"] = history["VIX_Return_Pct_Lag1"].fillna(history.get("VIX_Return_Pct", 0.0))

    if "SP500_Direction" not in history.columns and "SP500_Daily_Return_Pct" in history.columns:
        history["SP500_Direction"] = np.where(history["SP500_Daily_Return_Pct"] > 0, 1, 0)

    history["Surprise_Direction"] = np.select(
        [
            history["CPI_Surprise_Pct"] < 0,
            history["CPI_Surprise_Pct"].eq(0),
            history["CPI_Surprise_Pct"] > 0,
        ],
        [0, 1, 2],
        default=1,
    )
    return history


def _load_pickle_model(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    loaded_formats: dict[str, str] = {}

    try:
        if os.path.exists(MODEL_PATHS["surprise_model_json"]):
            from xgboost import XGBRegressor

            model1 = XGBRegressor()
            model1.load_model(MODEL_PATHS["surprise_model_json"])
            loaded_formats["surprise_model"] = "json"
        elif os.path.exists(MODEL_PATHS["surprise_model_pkl"]):
            model1 = _load_pickle_model(MODEL_PATHS["surprise_model_pkl"])
            loaded_formats["surprise_model"] = "pickle"
        else:
            return {}

        if os.path.exists(MODEL_PATHS["direction_model_json"]):
            from xgboost import XGBClassifier

            model2 = XGBClassifier()
            model2.load_model(MODEL_PATHS["direction_model_json"])
            loaded_formats["direction_model"] = "json"
        elif os.path.exists(MODEL_PATHS["direction_model_pkl"]):
            model2 = _load_pickle_model(MODEL_PATHS["direction_model_pkl"])
            loaded_formats["direction_model"] = "pickle"
        else:
            return {}

        if os.path.exists(MODEL_PATHS["spx_model_joblib"]):
            model3 = joblib.load(MODEL_PATHS["spx_model_joblib"])
            loaded_formats["spx_model"] = "joblib"
        elif os.path.exists(MODEL_PATHS["spx_model_pkl"]):
            model3 = _load_pickle_model(MODEL_PATHS["spx_model_pkl"])
            loaded_formats["spx_model"] = "pickle"
        else:
            return {}

        native_formats = {
            "surprise_model": "json",
            "direction_model": "json",
            "spx_model": "joblib",
        }
        unique_formats = set(loaded_formats.values())
        if loaded_formats == native_formats:
            model_format = "native"
        else:
            model_format = unique_formats.pop() if len(unique_formats) == 1 else "mixed"
        return {
            "surprise_model": model1,
            "direction_model": model2,
            "spx_model": model3,
            "format": model_format,
            "formats": loaded_formats,
        }
    except Exception:
        load_models.cache_clear()
        return {}


def latest_context(recent_data: pd.DataFrame) -> pd.Series:
    return recent_data.sort_values("Release_Date").iloc[-1]


def model_feature_names(model: Any, include_surprise: bool) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    if hasattr(model, "get_booster"):
        booster_names = model.get_booster().feature_names
        if booster_names:
            return [str(name) for name in booster_names]

    default = FEATURE_ORDER if include_surprise else MODEL_1_2_FEATURES
    return default


def align_features(row: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = row.copy()
    if "VIX_Return_Pct" in feature_names and "VIX_Return_Pct" not in aligned.columns:
        aligned["VIX_Return_Pct"] = aligned["VIX_Return_Pct_Lag1"]
    if "VIX_Return_Pct_Lag1" in feature_names and "VIX_Return_Pct_Lag1" not in aligned.columns:
        aligned["VIX_Return_Pct_Lag1"] = aligned["VIX_Return_Pct"]
    for name in feature_names:
        if name not in aligned.columns:
            aligned[name] = 0.0
    return aligned[feature_names].fillna(0.0)


def build_feature_row(
    forecast_cpi: float,
    recent_data: pd.DataFrame,
    actual_cpi: float | None = None,
    surprise_override: float | None = None,
) -> pd.DataFrame:
    df_sorted = prepare_history(recent_data)
    latest = latest_context(df_sorted)
    baseline_actual = float(latest.get("CPI_YoY", forecast_cpi))

    if surprise_override is not None:
        surprise = float(surprise_override)
    elif actual_cpi is not None:
        surprise = float(actual_cpi) - float(forecast_cpi)
    else:
        surprise = float(forecast_cpi) - baseline_actual

    regime_encoded = latest.get("Inflation_Regime_Encoded", None)
    if pd.isna(regime_encoded):
        regime_encoded = encode_regime(latest.get("Inflation_Regime", "Low"))

    vix_lag = latest.get("VIX_Return_Pct_Lag1", None)
    if pd.isna(vix_lag):
        if len(df_sorted) > 1:
            vix_lag = df_sorted.iloc[-2].get("VIX_Return_Pct", 0.0)
        else:
            vix_lag = latest.get("VIX_Return_Pct", 0.0)

    row = {
        "CPI_Surprise_Pct": float(surprise),
        "VIX_Return_Pct_Lag1": float(vix_lag),
        "VIX_Return_Pct": float(latest.get("VIX_Return_Pct", vix_lag)),
        "SP500_Return_Lag1": float(latest.get("SP500_Daily_Return_Pct", latest.get("SP500_Return_Lag1", 0.0))),
        "FedFunds_Rate": float(latest.get("FedFunds_Rate", 0.0)),
        "FedFunds_Change_3M": float(latest.get("FedFunds_Change_3M", 0.0)),
        "Inflation_Regime_Encoded": int(regime_encoded),
    }
    return pd.DataFrame([row])


def class_label(model: Any, class_index: int, labels: dict[int, str]) -> str:
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > class_index:
        raw_value = classes[class_index]
        if isinstance(raw_value, str):
            return raw_value
        try:
            return labels.get(int(raw_value), str(raw_value))
        except (TypeError, ValueError):
            return str(raw_value)
    return labels.get(class_index, str(class_index))


def feature_importance(model: Any, feature_row: pd.DataFrame, feature_names: list[str]) -> list[dict[str, float | str]]:
    values = None
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)[0])
    elif hasattr(model, "feature_importances_"):
        values = np.abs(np.asarray(model.feature_importances_))

    if values is None or len(values) != len(feature_names):
        return []

    aligned = align_features(feature_row, feature_names)
    total = float(np.sum(values)) or 1.0
    items = []
    for index, name in enumerate(feature_names):
        items.append(
            {
                "name": name,
                "value": float(aligned[name].iloc[0]),
                "importance": float(values[index] / total),
            }
        )
    return sorted(items, key=lambda item: item["importance"], reverse=True)[:3]


def find_similar_events(recent_data: pd.DataFrame, actual_surprise: float, tolerance: float = 0.10) -> pd.DataFrame:
    history = prepare_history(recent_data)
    latest = latest_context(history)
    current_regime = int(latest.get("Inflation_Regime_Encoded", 0))
    mask = (
        history["Inflation_Regime_Encoded"].eq(current_regime)
        & history["CPI_Surprise_Pct"].sub(actual_surprise).abs().le(tolerance)
    )
    return history.loc[mask].copy()


def fallback_prediction(actual_cpi: float, forecast_cpi: float, recent_data: pd.DataFrame) -> dict[str, Any]:
    actual_surprise = float(actual_cpi) - float(forecast_cpi)
    direction = "Above" if actual_surprise > 0 else "Below" if actual_surprise < 0 else "Match"
    spx_direction = "Down" if actual_surprise > 0 else "Up" if actual_surprise < 0 else "Neutral"
    similar = find_similar_events(recent_data, actual_surprise)

    return {
        "actual_surprise": actual_surprise,
        "predicted_surprise": actual_surprise,
        "surprise_direction": direction,
        "surprise_confidence": 0.0,
        "spx_direction": spx_direction,
        "spx_probability": 0.0,
        "top_features": [],
        "similar_events": similar,
        "similar_events_count": len(similar),
        "similar_events_up": int((similar["SP500_Direction"] == 1).sum()) if len(similar) else 0,
        "similar_events_down": int((similar["SP500_Direction"] == 0).sum()) if len(similar) else 0,
        "model_accuracy": {"surprise_direction_acc": None, "spx_direction_acc": None},
        "used_fallback": True,
    }


def evaluate_models(models: dict[str, Any], recent_data: pd.DataFrame) -> dict[str, float | None]:
    try:
        history = prepare_history(recent_data).dropna(subset=["CPI_Surprise_Pct"])
        direction_model = models["direction_model"]
        spx_model = models["spx_model"]
        direction_features = model_feature_names(direction_model, include_surprise=False)
        spx_features = model_feature_names(spx_model, include_surprise=True)

        direction_pred = direction_model.predict(align_features(history, direction_features))
        spx_pred = spx_model.predict(align_features(history, spx_features))

        direction_actual = history["Surprise_Direction"].astype(int).to_numpy()
        spx_actual = history["SP500_Direction"].astype(int).to_numpy()

        direction_pred = np.asarray(direction_pred).astype(int)
        spx_pred = np.asarray(spx_pred).astype(int)

        return {
            "surprise_direction_acc": float(np.mean(direction_pred == direction_actual)),
            "spx_direction_acc": float(np.mean(spx_pred == spx_actual)),
        }
    except Exception:
        return {"surprise_direction_acc": None, "spx_direction_acc": None}


def predict(actual_cpi: float, forecast_cpi: float, recent_data: pd.DataFrame | None = None) -> dict[str, Any]:
    if recent_data is None:
        recent_data = load_data()
    else:
        recent_data = prepare_history(recent_data)

    models = load_models()
    if not models:
        return fallback_prediction(actual_cpi, forecast_cpi, recent_data)

    actual_surprise = float(actual_cpi) - float(forecast_cpi)
    pre_release_row = build_feature_row(forecast_cpi=forecast_cpi, recent_data=recent_data)
    actual_row = build_feature_row(
        forecast_cpi=forecast_cpi,
        actual_cpi=actual_cpi,
        recent_data=recent_data,
        surprise_override=actual_surprise,
    )

    model1 = models["surprise_model"]
    model2 = models["direction_model"]
    model3 = models["spx_model"]

    model1_features = model_feature_names(model1, include_surprise=False)
    model2_features = model_feature_names(model2, include_surprise=False)
    model3_features = model_feature_names(model3, include_surprise=True)

    predicted_surprise = float(model1.predict(align_features(pre_release_row, model1_features))[0])

    direction_proba = np.asarray(model2.predict_proba(align_features(pre_release_row, model2_features))[0])
    direction_index = int(direction_proba.argmax())
    surprise_direction = class_label(model2, direction_index, SURPRISE_CLASS_LABELS)

    spx_proba = np.asarray(model3.predict_proba(align_features(actual_row, model3_features))[0])
    spx_index = int(spx_proba.argmax())
    spx_direction = class_label(model3, spx_index, SPX_CLASS_LABELS)

    similar = find_similar_events(recent_data, actual_surprise)
    sim_count = len(similar)
    sim_up = int((similar["SP500_Direction"] == 1).sum()) if sim_count else 0

    return {
        "actual_surprise": actual_surprise,
        "predicted_surprise": predicted_surprise,
        "surprise_direction": str(surprise_direction),
        "surprise_confidence": float(direction_proba.max() * 100),
        "spx_direction": str(spx_direction),
        "spx_probability": float(spx_proba.max() * 100),
        "top_features": feature_importance(model3, actual_row, model3_features),
        "similar_events": similar,
        "similar_events_count": sim_count,
        "similar_events_up": sim_up,
        "similar_events_down": sim_count - sim_up,
        "model_accuracy": evaluate_models(models, recent_data),
        "used_fallback": False,
        "model_format": models.get("format", "unknown"),
    }
