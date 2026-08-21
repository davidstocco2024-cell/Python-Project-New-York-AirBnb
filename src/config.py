from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# El CSV original, en su lugar de siempre. Nada de rutas de Colab.
RAW_FILE = ROOT / "datasets.csv"

DATA_PROCESSED = ROOT / "data" / "processed"
CLEAN_FILE = DATA_PROCESSED / "listings_clean.parquet"

# Modelo serializado que consume la app de Streamlit
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "price_model.joblib"

# Datos geograficos externos (los baja scripts/fetch_geo_data.py)
GEO_DIR = ROOT / "data" / "geo"
SUBWAY_FILE = GEO_DIR / "subway.csv"
POI_FILE = GEO_DIR / "poi.csv"

# Centinelas encontrados auditando el CSV.
# OJO: 'New ' trae un espacio al final en el archivo original.
RATING_SENTINELS = {"No rating", "New"}
BATHS_SENTINELS = {"Not specified"}
BEDROOMS_STUDIO = "Studio"  # un estudio es 0 recamaras, no un nulo

# Local Law 18 de NYC: solo regula estancias MENORES a 30 noches.
# Exigir 30+ es la via legal para operar sin registrarse.
LL18_LOOPHOLE_NIGHTS = 30

# Puntos de referencia (lat, lon) para features de distancia. Costo de API: cero.
LANDMARKS = {
    "times_square": (40.7580, -73.9855),
    "central_park_s": (40.7660, -73.9776),
    "wall_street": (40.7069, -74.0113),
    "empire_state": (40.7484, -73.9857),
    "jfk_airport": (40.6413, -73.7781),
    "williamsburg": (40.7081, -73.9571),
}

# Corte duro solo para precios imposibles. El recorte fino va por IQR en log.
PRICE_HARD_MIN = 10
PRICE_HARD_MAX = 100_000

RANDOM_STATE = 42