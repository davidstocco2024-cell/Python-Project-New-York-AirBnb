"""
Paso 1: limpieza. Lee datasets.csv de la raiz del repo y guarda el parquet.

    python scripts/01_clean.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean          # noqa: E402
from src.config import CLEAN_FILE, RAW_FILE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

df = clean(RAW_FILE)
CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(CLEAN_FILE, index=False)

print(f"\nguardado en {CLEAN_FILE.relative_to(CLEAN_FILE.parents[2])}  shape={df.shape}")
print("\ncolumnas numericas rescatadas (eran texto en el original):")
print(df[["rating", "bedrooms", "baths", "beds"]].describe().round(2).to_string())
print("\nlicense_status:")
print(df["license_status"].value_counts().to_string())