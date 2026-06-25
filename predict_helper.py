from __future__ import annotations

import os
import pickle
import sys
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "cpi_spx.csv")

MODEL_PATHS = {
    "surprise_model": os.path.join(MODEL_DIR, "model1_surprise_regression.pkl"),
    "direction_model": os.path.join(MODEL_DIR, "model2_direction_classifier.pkl"),
    "spx_model": os.path.join(MODEL_DIR, "model3_spx_classifier.pkl"),
}

FEATURE_ORDER = [
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


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    history = df.copy()
    history["Release_Date"] = pd.to_datetime(history["Release_Date"])

    if "Inflation_Regime_Encoded" not in history.columns:
        history["Inflation_Regime_Encoded"] = np.select(
            [
                history["Inflation_Regime"].astype(str).str.lower().eq("high"),
                history["Inflation_Regime"].astype(str).str.lower().eq("medium"),
            ],
            [2, 1],
            default=0,
        )

    if "SP500_Direction" not in history.columns:
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
    return history.sort_values("Release_Date").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    if not all(os.path.exists(path) for path in MODEL_PATHS.values()):
        return {}

    models = {}
    for key, path in MODEL_PATHS.items():
        with open(path, "rb") as file:
            models[key] = pickle.load(file)
    return models


def latest_context(recent_data: pd.DataFrame) -> pd.Series:
    return recent_data.sort_values("Release_Date").iloc[-1]


def encode_regime(value: Any) -> int:
    text = str(value).lower()
    if text == "high":
        return 2
    if text == "medium":
        return 1
    return 0


def build_feature_row(
    forecast_cpi: float,
    recent_data: pd.DataFrame,
    actual_cpi: float | None = None,
    surprise_override: float | None = None,
) -> pd.DataFrame:
    latest = latest_context(recent_data)
    baseline_actual = float(latest.get("CPI_YoY", forecast_cpi))

    if surprise_override is not None:
        surprise = float(surprise_override)
    elif actual_cpi is not None:
        surprise = float(actual_cpi) - float(forecast_cpi)
    else:
        # Pre-release mode mirrors the original helper: estimate the surprise
        # relative to the latest known CPI YoY when actual CPI is unavailable.
        surprise = float(forecast_cpi) - baseline_actual

    regime_encoded = latest.get("Inflation_Regime_Encoded", None)
    if pd.isna(regime_encoded):
        regime_encoded = encode_regime(latest.get("Inflation_Regime", "Low"))

    row = {
        "CPI_Surprise_Pct": surprise,
        "VIX_Return_Pct": float(latest.get("VIX_Return_Pct", 0.0)),
        "SP500_Return_Lag1": float(latest.get("SP500_Daily_Return_Pct", 0.0)),
        "FedFunds_Rate": float(latest.get("FedFunds_Rate", 0.0)),
        "FedFunds_Change_3M": float(latest.get("FedFunds_Change_3M", 0.0)),
        "Inflation_Regime_Encoded": int(regime_encoded),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


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


def feature_importance(model: Any, feature_row: pd.DataFrame) -> list[dict[str, float | str]]:
    values = None
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)[0])
    elif hasattr(model, "feature_importances_"):
        values = np.abs(np.asarray(model.feature_importances_))

    if values is None or len(values) != len(FEATURE_ORDER):
        return []

    total = float(np.sum(values)) or 1.0
    items = []
    for index, name in enumerate(FEATURE_ORDER):
        items.append(
            {
                "name": name,
                "value": float(feature_row[name].iloc[0]),
                "importance": float(values[index] / total),
            }
        )
    return sorted(items, key=lambda item: item["importance"], reverse=True)[:3]


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


def find_similar_events(recent_data: pd.DataFrame, actual_surprise: float, tolerance: float = 0.10) -> pd.DataFrame:
    history = prepare_history(recent_data)
    latest = latest_context(history)
    current_regime = int(latest.get("Inflation_Regime_Encoded", 0))
    mask = (
        history["Inflation_Regime_Encoded"].eq(current_regime)
        & history["CPI_Surprise_Pct"].sub(actual_surprise).abs().le(tolerance)
    )
    return history.loc[mask].copy()


def evaluate_models(models: dict[str, Any], recent_data: pd.DataFrame) -> dict[str, float | None]:
    try:
        history = prepare_history(recent_data).dropna(subset=MODEL_1_2_FEATURES + ["CPI_Surprise_Pct"])
        direction_model = models["direction_model"]
        spx_model = models["spx_model"]

        direction_pred = direction_model.predict(history[MODEL_1_2_FEATURES])
        spx_pred = spx_model.predict(history[FEATURE_ORDER])

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

    predicted_surprise = float(model1.predict(pre_release_row[MODEL_1_2_FEATURES])[0])

    direction_proba = np.asarray(model2.predict_proba(pre_release_row[MODEL_1_2_FEATURES])[0])
    direction_index = int(direction_proba.argmax())
    surprise_direction = class_label(model2, direction_index, SURPRISE_CLASS_LABELS)

    spx_proba = np.asarray(model3.predict_proba(actual_row[FEATURE_ORDER])[0])
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
        "top_features": feature_importance(model3, actual_row),
        "similar_events": similar,
        "similar_events_count": sim_count,
        "similar_events_up": sim_up,
        "similar_events_down": sim_count - sim_up,
        "model_accuracy": evaluate_models(models, recent_data),
        "used_fallback": False,
    }
