"""
Baja datos geograficos GRATIS para enriquecer los listings.

Sustituye a Google Places API: con Nearby Search en tier Pro serian
~$32 USD por cada 1,000 llamadas, o sea ~$650 por los 20,600 listings.
Overpass y NYC Open Data cuestan cero y no tienen tope de uso razonable.

Correr una sola vez; el resultado se cachea en data/geo/.

    python scripts/fetch_geo_data.py

NOTA: requiere salida a internet hacia overpass-api.de.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GEO_DIR  # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"

# Bounding box de los cinco boroughs (sur, oeste, norte, este)
NYC_BBOX = "40.49,-74.27,40.92,-73.68"

QUERIES = {
    "subway": f"""
        [out:json][timeout:180];
        (
          node["railway"="subway_entrance"]({NYC_BBOX});
          node["station"="subway"]({NYC_BBOX});
        );
        out body;
    """,
    # POIs que mueven el valor de una renta corta
    "poi": f"""
        [out:json][timeout:180];
        (
          node["amenity"~"restaurant|cafe|bar|pub"]({NYC_BBOX});
          node["shop"~"supermarket|convenience"]({NYC_BBOX});
          node["leisure"="park"]({NYC_BBOX});
        );
        out body;
    """,
}


def fetch_overpass(query: str, retries: int = 3) -> pd.DataFrame:
    """Overpass devuelve 429 seguido. Reintento con backoff, sin drama."""
    for attempt in range(retries):
        r = requests.post(OVERPASS, data={"data": query}, timeout=300)
        if r.status_code == 200:
            elements = r.json().get("elements", [])
            return pd.DataFrame(
                [
                    {"latitude": e["lat"], "longitude": e["lon"],
                     "name": e.get("tags", {}).get("name")}
                    for e in elements
                    if "lat" in e and "lon" in e
                ]
            )
        wait = 10 * (attempt + 1)
        print(f"  status {r.status_code}, reintento en {wait}s")
        time.sleep(wait)
    raise RuntimeError("Overpass no respondio despues de varios intentos")


def main() -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    for name, query in QUERIES.items():
        out = GEO_DIR / f"{name}.csv"
        if out.exists():
            print(f"{name}: ya existe, se omite")
            continue
        print(f"{name}: descargando...")
        df = fetch_overpass(query)
        df.to_csv(out, index=False)
        print(f"  {len(df):,} puntos -> {out}")
        time.sleep(5)  # cortesia con el servidor publico


if __name__ == "__main__":
    main()