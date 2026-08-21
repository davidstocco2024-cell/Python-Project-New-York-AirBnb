from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

from .config import RANDOM_STATE

log = logging.getLogger(__name__)

# Cualquier columna derivada del precio es leakage. Se bloquea explicitamente.
BANNED = {"price", "log_price", "price per bed", "price_per_bed", "price_per_person"}

CATEGORICAL = ["neighbourhood_group", "room_type", "license_status"]


def assert_no_leakage(features: list[str]) -> None:
    """Falla ruidosamente si alguna feature huele a precio."""
    bad = [f for f in features if f in BANNED or "price" in f.lower()]
    if bad:
        raise ValueError(f"leakage detectado en features: {bad}")


def prepare(df: pd.DataFrame, feature_cols: list[str]):
    assert_no_leakage(feature_cols)
    d = df[df["is_price_outlier"] == 0].copy()

    X = d[feature_cols].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.log(d["price"].to_numpy())
    groups = d["neighbourhood"].to_numpy()
    return X, y, groups


def evaluate(X, y, groups, label: str, n_splits: int = 5) -> dict:
    """Compara split aleatorio vs split espacial. La brecha es el hallazgo."""
    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )

    results = {}
    for scheme, cv, g in [
        ("aleatorio", KFold(n_splits, shuffle=True, random_state=RANDOM_STATE), None),
        ("por barrio", GroupKFold(n_splits), groups),
    ]:
        pred = cross_val_predict(model, X, y, cv=cv, groups=g)
        results[scheme] = {
            "r2_log": r2_score(y, pred),
            # MAE en dolares reales, que es lo que le importa a un host
            "mae_usd": mean_absolute_error(np.exp(y), np.exp(pred)),
        }
        log.info(
            "%-22s %-11s R2=%.3f  MAE=$%.0f",
            label, scheme, results[scheme]["r2_log"], results[scheme]["mae_usd"],
        )
    return results


def fit_full(X, y) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X, y)
    return model


def importance_table(model: LGBMRegressor, X: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": X.columns, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )