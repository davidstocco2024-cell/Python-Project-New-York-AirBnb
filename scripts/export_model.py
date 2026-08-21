"""
Paso 4: entrena el modelo final y lo guarda para la app.
Requiere haber corrido 01_clean.py.

    python scripts/04_export_model.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_FILE, MODEL_FILE, POI_FILE, SUBWAY_FILE  # noqa: E402
from src.features import build_features                                # noqa: E402
from src.modeling import CATEGORICAL, fit_full, prepare                # noqa: E402
from src.predict import Bundle                                         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_optional(p):
    return pd.read_csv(p) if p.exists() else None


df = build_features(
    pd.read_parquet(CLEAN_FILE),
    subway=load_optional(SUBWAY_FILE),
    pois=load_optional(POI_FILE),
)

# Mismo set C de 03_train.py, pero SIN las columnas que la app no puede pedirle
# al usuario de forma razonable (reviews historicos, disponibilidad futura).
# Un host que aun no publica no tiene esos datos, asi que pedirlos seria
# entrenar con informacion que en produccion no existe.
FEATURES = [
    "latitude", "longitude", "minimum_nights",
    "neighbourhood_group", "room_type", "license_status",
    "rating", "has_rating", "bedrooms", "baths", "beds", "is_studio",
    "is_registered", "uses_30night_loophole", "is_str_legal",
    "is_multi_listing_host", "log_host_listings", "beds_per_bedroom",
    "listings_in_neighbourhood",
] + [c for c in df.columns if c.startswith(("dist_", "n_subway_", "n_poi_"))]

X, y, _ = prepare(df, FEATURES)
model = fit_full(X, y)

d = df[df["is_price_outlier"] == 0]
bundle = Bundle(
    model=model,
    feature_cols=list(X.columns),
    categories={c: list(X[c].cat.categories) for c in CATEGORICAL if c in X.columns},
    reference=d[["latitude", "longitude", "price", "room_type",
                 "neighbourhood", "neighbourhood_group", "license_status"]].reset_index(drop=True),
)
bundle.save(MODEL_FILE)
print(f"\nmodelo guardado en {MODEL_FILE}")
print(f"features: {len(bundle.feature_cols)} | comparables: {len(bundle.reference):,}")