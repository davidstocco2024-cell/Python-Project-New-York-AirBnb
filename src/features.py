
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import LANDMARKS, LL18_LOOPHOLE_NIGHTS

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia sobre la esfera. Vectorizada, sin API de por medio."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_landmark_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Distancia en km a puntos fijos de referencia. Costo de API: cero."""
    df = df.copy()
    for name, (lat, lon) in LANDMARKS.items():
        df[f"dist_{name}_km"] = haversine_km(
            df["latitude"].to_numpy(), df["longitude"].to_numpy(), lat, lon
        )
    return df


def add_poi_features(
    df: pd.DataFrame, poi: pd.DataFrame | None, prefix: str, radii_m=(500, 1000)
) -> pd.DataFrame:
    """
    Distancia al POI mas cercano + conteo dentro de radios, con BallTree
    en metrica haversine. 20k listings contra 500 estaciones corre en
    menos de un segundo; con Places API serian ~$500 dolares.

    `poi` debe traer columnas latitude / longitude. Si es None, se omite
    (para poder correr el pipeline sin haber bajado los datos externos).
    """
    df = df.copy()
    if poi is None or poi.empty:
        log.warning("sin datos de '%s', se omiten esas features", prefix)
        return df

    tree = BallTree(np.radians(poi[["latitude", "longitude"]].to_numpy()), metric="haversine")
    pts = np.radians(df[["latitude", "longitude"]].to_numpy())

    dist, _ = tree.query(pts, k=1)
    df[f"dist_{prefix}_km"] = dist[:, 0] * EARTH_RADIUS_KM

    for r in radii_m:
        counts = tree.query_radius(pts, r=(r / 1000) / EARTH_RADIUS_KM, count_only=True)
        df[f"n_{prefix}_{r}m"] = counts.astype("int16")
    return df


def add_ll18_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uses_30night_loophole"] = (
        df["minimum_nights"] >= LL18_LOOPHOLE_NIGHTS
    ).astype("int8")
    df["is_str_legal"] = (
        (df["minimum_nights"] < LL18_LOOPHOLE_NIGHTS) & (df["is_registered"] == 1)
    ).astype("int8")
    return df


def add_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """Señales de host profesional vs casero. host_id sobrevivio intacto a Excel."""
    df = df.copy()
    df["is_multi_listing_host"] = (df["calculated_host_listings_count"] > 1).astype("int8")
    df["log_host_listings"] = np.log1p(df["calculated_host_listings_count"])
    df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, 1)
    df["days_since_review"] = (
        pd.Timestamp("2024-01-05") - df["last_review"]
    ).dt.days
    return df


def add_neighbourhood_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estadisticos del barrio que NO tocan price: densidad de oferta.
    (Un target-encoding sobre price iria aqui, pero tiene que calcularse
    dentro del fold de CV, no antes. Por eso no esta en este modulo.)
    """
    df = df.copy()
    counts = df.groupby("neighbourhood", observed=True)["neighbourhood"].transform("size")
    df["listings_in_neighbourhood"] = counts.astype("int32")
    return df


def build_features(df: pd.DataFrame, subway: pd.DataFrame | None = None,
                   pois: pd.DataFrame | None = None) -> pd.DataFrame:
    out = (
        df.pipe(add_landmark_distances)
        .pipe(add_poi_features, subway, "subway")
        .pipe(add_poi_features, pois, "poi", radii_m=(300, 800))
        .pipe(add_ll18_features)
        .pipe(add_host_features)
        .pipe(add_neighbourhood_stats)
    )
    log.info("features construidas: %d columnas", out.shape[1])
    return out