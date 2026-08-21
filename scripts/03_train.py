"""
Paso 3: entrena y compara tres conjuntos de features.
Requiere haber corrido 01_clean.py antes.

    python scripts/03_train.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_FILE, POI_FILE, SUBWAY_FILE  # noqa: E402
from src.features import build_features                    # noqa: E402
from src.modeling import evaluate, fit_full, importance_table, prepare  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_optional(path):
    """Los datos geo son opcionales: si no se bajaron, el pipeline sigue sin ellos."""
    return pd.read_csv(path) if path.exists() else None


df = build_features(
    pd.read_parquet(CLEAN_FILE),
    subway=load_optional(SUBWAY_FILE),
    pois=load_optional(POI_FILE),
)

# A) lo que el notebook original tenia disponible
BASE = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365", "beds", "neighbourhood_group", "room_type",
]

# B) + columnas que la limpieza rescata (eran texto y se perdian)
RESCUED = BASE + [
    "rating", "has_rating", "is_new_listing", "bedrooms", "baths",
    "is_studio", "license_status", "is_registered",
]

# C) + geografia y variables regulatorias
FULL = RESCUED + [c for c in df.columns if c.startswith(("dist_", "n_subway_", "n_poi_"))] + [
    "uses_30night_loophole", "is_str_legal", "is_multi_listing_host",
    "log_host_listings", "beds_per_bedroom", "days_since_review",
    "listings_in_neighbourhood",
]

for label, cols in [
    ("A) notebook original", BASE),
    ("B) + rescatadas", RESCUED),
    ("C) + geo/LL18", FULL),
]:
    X, y, groups = prepare(df, cols)
    evaluate(X, y, groups, label)
    print()

X, y, groups = prepare(df, FULL)
model = fit_full(X, y)
print("TOP FEATURES (gain):")
print(importance_table(model, X).to_string(index=False))