"""
Price Advisor — NYC Airbnb 2024

    streamlit run app.py

Requiere haber corrido antes:
    python scripts/01_clean.py
    python scripts/04_export_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODEL_FILE  # noqa: E402
from src.predict import (  # noqa: E402
    Bundle,
    build_input_row,
    comparables,
    explain,
    percentile_of,
    predict_price,
)

st.set_page_config(page_title="NYC Airbnb Price Advisor", layout="wide")


@st.cache_resource
def load_bundle() -> Bundle:
    if not MODEL_FILE.exists():
        st.error(
            "No encuentro el modelo. Corre primero:\n\n"
            "```\npython scripts/01_clean.py\npython scripts/04_export_model.py\n```"
        )
        st.stop()
    return Bundle.load(MODEL_FILE)


bundle = load_bundle()
ref = bundle.reference


@st.cache_data
def neighbourhood_index() -> pd.DataFrame:
    """Centroide y conteo por barrio, para prellenar coordenadas."""
    return (
        ref.groupby(["neighbourhood_group", "neighbourhood"], observed=True)
        .agg(lat=("latitude", "median"), lon=("longitude", "median"), n=("price", "size"))
        .reset_index()
    )


idx = neighbourhood_index()

st.title("NYC Airbnb Price Advisor")
st.caption(
    "Precio sugerido a partir de un LightGBM sobre log(price), validado con "
    "GroupKFold por barrio (R² = 0.618, MAE ≈ $60). Snapshot de enero 2024."
)

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Tu listing")

    borough = st.selectbox("Borough", sorted(idx["neighbourhood_group"].unique()))
    barrios = idx[idx["neighbourhood_group"] == borough].sort_values("n", ascending=False)
    barrio = st.selectbox("Barrio", barrios["neighbourhood"].tolist())

    fila = barrios[barrios["neighbourhood"] == barrio].iloc[0]
    st.caption(f"{int(fila['n'])} listings en este barrio")

    with st.expander("Ajustar coordenadas"):
        lat = st.number_input("Latitud", value=float(fila["lat"]), format="%.5f")
        lon = st.number_input("Longitud", value=float(fila["lon"]), format="%.5f")

    st.divider()
    room_type = st.selectbox("Tipo", sorted(ref["room_type"].unique()))
    c1, c2 = st.columns(2)
    bedrooms = c1.number_input("Recámaras", 0, 10, 1, help="0 = estudio")
    beds = c2.number_input("Camas", 1, 20, 1)
    baths = c1.number_input("Baños", 0.0, 10.0, 1.0, step=0.5)
    rating = c2.number_input("Rating", 1.0, 5.0, 4.8, step=0.1)

    st.divider()
    st.subheader("Situación regulatoria")
    license_status = st.selectbox(
        "Licencia",
        ["none", "registered", "exempt"],
        format_func={
            "none": "Sin licencia",
            "registered": "Registrado (OSE-STRREG)",
            "exempt": "Exento",
        }.get,
    )
    minimum_nights = st.number_input("Noches mínimas", 1, 365, 30)
    host_listings = st.number_input("Listings del host", 1, 200, 1)

    radius = st.slider("Radio de comparables (km)", 0.3, 3.0, 1.0, 0.1)

# ------------------------------------------------------------- prediccion
X = build_input_row(
    latitude=lat,
    longitude=lon,
    neighbourhood_group=borough,
    neighbourhood=barrio,
    room_type=room_type,
    bedrooms=bedrooms,
    beds=beds,
    baths=baths,
    minimum_nights=minimum_nights,
    license_status=license_status,
    listings_in_neighbourhood=int(fila["n"]),
    overrides={"rating": rating, "calculated_host_listings_count": host_listings},
)

price = predict_price(bundle, X)
comps = comparables(bundle, lat, lon, radius, room_type)
pct = percentile_of(price, comps)

# ------------------------------------------------------------------ salida
k1, k2, k3, k4 = st.columns(4)
k1.metric("Precio sugerido", f"${price:,.0f}", help="Por noche")
k2.metric("Comparables", f"{len(comps):,}", help=f"En {radius} km")
if not comps.empty:
    k3.metric("Mediana de la zona", f"${comps['price'].median():,.0f}")
    k4.metric("Percentil", f"{pct:.0f}º" if pct is not None else "—")

if len(comps) < 20:
    st.warning(
        f"Solo {len(comps)} comparables en {radius} km. El percentil no es "
        "confiable con esa muestra; amplía el radio."
    )

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("Comparables en la zona")
    if comps.empty:
        st.info("Sin listings en el radio seleccionado.")
    else:
        mapa = comps[["latitude", "longitude"]].rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        st.map(pd.concat([mapa, pd.DataFrame([{"lat": lat, "lon": lon}])]), size=20)

        st.caption("Distribución de precios de los comparables")
        hist = np.histogram(comps["price"], bins=30)
        st.bar_chart(
            pd.DataFrame({"listings": hist[0]}, index=hist[1][:-1].round(0).astype(int))
        )

with right:
    st.subheader("Qué mueve el precio")
    st.caption(
        "Contribuciones exactas por feature (TreeSHAP). En % sobre el precio base."
    )
    exp = explain(bundle, X, top=8)
    exp_display = exp[["feature", "efecto_pct"]].copy()
    exp_display["efecto_pct"] = exp_display["efecto_pct"].round(1)
    st.dataframe(
        exp_display.rename(columns={"feature": "Feature", "efecto_pct": "Efecto %"}),
        hide_index=True,
        use_container_width=True,
    )

# --------------------------------------------------- escenario regulatorio
st.divider()
st.subheader("Escenario: ¿cuánto cambia si te registras?")
st.caption(
    "La Local Law 18 solo regula estancias menores a 30 noches. En el dataset, "
    "99.96% de los listings sin licencia exigen 30+ noches, y los registrados "
    "reciben ~8x más reservas al año."
)

escenarios = []
for lic, mn, label in [
    ("none", 30, "Sin licencia, mínimo 30 noches"),
    ("registered", 2, "Registrado, corta estancia"),
    ("exempt", 2, "Exento, corta estancia"),
]:
    Xs = build_input_row(
        latitude=lat, longitude=lon, neighbourhood_group=borough, neighbourhood=barrio,
        room_type=room_type, bedrooms=bedrooms, beds=beds, baths=baths,
        minimum_nights=mn, license_status=lic, listings_in_neighbourhood=int(fila["n"]),
        overrides={"rating": rating, "calculated_host_listings_count": host_listings},
    )
    escenarios.append({"Escenario": label, "Precio/noche": f"${predict_price(bundle, Xs):,.0f}"})

st.dataframe(pd.DataFrame(escenarios), hide_index=True, use_container_width=True)
st.caption(
    "⚠️ Compara precio por noche, no ingreso total. Un listing de 30 noches "
    "mínimas y uno de corta estancia tienen ocupación muy distinta, y el modelo "
    "no predice ocupación — el dataset es una foto estática sin serie de tiempo."
)

with st.expander("Limitaciones del modelo"):
    st.markdown(
        """
- **Snapshot de enero 2024.** No hay componente temporal, así que no predice
  estacionalidad ni tendencia. Los precios de hoy serán distintos.
- **Predice precio pedido, no precio óptimo.** El modelo aprende qué cobran
  hosts parecidos, no qué maximiza ingreso.
- **R² de 0.618** con validación por barrio: la ubicación exacta, las fotos y
  la calidad del inmueble explican mucho de lo que falta.
- **Correlación, no causalidad.** El escenario de registro muestra qué cobran
  listings registrados similares, no qué pasaría si tú te registras.
- **38.8% de los `id` del dataset venían dañados por Excel** y esas filas
  difieren sistemáticamente del resto (ver `ANALYSIS.md`).
        """
    )