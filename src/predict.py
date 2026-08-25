from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import EARTH_RADIUS_KM, add_landmark_distances, add_ll18_features, haversine_km

# Valores por defecto para las columnas que el usuario no captura en la app.
# Son medianas del dataset; se documentan aqui para que no queden escondidas.
DEFAULTS = {
    "number_of_reviews": 5.0,
    "number_of_reviews_ltm": 3.0,
    "reviews_per_month": 0.5,
    "availability_365": 180,
    "calculated_host_listings_count": 1,
    "rating": 4.8,
    "has_rating": 1,
    "is_new_listing": 0,
    "days_since_review": 180.0,
}


@dataclass
class Bundle:
    """Todo lo que la app necesita para predecir, en un solo archivo."""

    model: object
    feature_cols: list[str]
    categories: dict[str, list]
    reference: pd.DataFrame  # comparables: lat, lon, price, room_type, neighbourhood

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Bundle":
        return joblib.load(path)


def build_input_row(
    *,
    latitude: float,
    longitude: float,
    neighbourhood_group: str,
    neighbourhood: str,
    room_type: str,
    bedrooms: float,
    beds: float,
    baths: float,
    minimum_nights: int,
    license_status: str,
    listings_in_neighbourhood: int,
    overrides: dict | None = None,
) -> pd.DataFrame:
    row = {
        "latitude": latitude,
        "longitude": longitude,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "bedrooms": float(bedrooms),
        "beds": float(beds),
        "baths": float(baths),
        "minimum_nights": int(minimum_nights),
        "license_status": license_status,
        "is_registered": int(license_status == "registered"),
        "is_studio": int(bedrooms == 0),
        "listings_in_neighbourhood": int(listings_in_neighbourhood),
        **DEFAULTS,
        **(overrides or {}),
    }
    df = pd.DataFrame([row])

    df = add_landmark_distances(df)
    df = add_ll18_features(df)
    df["is_multi_listing_host"] = int(df["calculated_host_listings_count"].iloc[0] > 1)
    df["log_host_listings"] = np.log1p(df["calculated_host_listings_count"])
    df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, 1)
    return df


def _align(X: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    X = X.copy()
    for col in bundle.feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[bundle.feature_cols]
    for col, cats in bundle.categories.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def predict_price(bundle: Bundle, X_raw: pd.DataFrame) -> float:
    X = _align(X_raw, bundle)
    return float(np.exp(bundle.model.predict(X)[0]))


def explain(bundle: Bundle, X_raw: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    X = _align(X_raw, bundle)
    contrib = bundle.model.predict(X, pred_contrib=True)[0]
    # la ultima posicion es el valor base (intercepto), no una feature
    out = pd.DataFrame({"feature": bundle.feature_cols, "contrib_log": contrib[:-1]})
    out["efecto_pct"] = (np.exp(out["contrib_log"]) - 1) * 100
    out["abs"] = out["contrib_log"].abs()
    return out.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


def comparables(
    bundle: Bundle, latitude: float, longitude: float, radius_km: float = 1.0,
    room_type: str | None = None,
) -> pd.DataFrame:
    """Listings reales dentro del radio. Es el 'contra que me comparo' del host."""
    ref = bundle.reference
    d = haversine_km(ref["latitude"].to_numpy(), ref["longitude"].to_numpy(), latitude, longitude)
    out = ref.loc[d <= radius_km].copy()
    out["dist_km"] = d[d <= radius_km]
    if room_type:
        same = out[out["room_type"] == room_type]
        if len(same) >= 15:  # solo filtra si quedan suficientes para un percentil util
            out = same
    return out.sort_values("dist_km")


def percentile_of(price: float, comps: pd.DataFrame) -> float | None:
    if comps.empty:
        return None
    return float((comps["price"] < price).mean() * 100)