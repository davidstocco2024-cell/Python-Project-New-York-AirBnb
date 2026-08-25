from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    BATHS_SENTINELS,
    BEDROOMS_STUDIO,
    PRICE_HARD_MAX,
    PRICE_HARD_MIN,
    RATING_SENTINELS,
)

log = logging.getLogger(__name__)


def load_raw(path) -> pd.DataFrame:
    """
    Carga el CSV crudo. id y host_id se leen como TEXTO a proposito:
    el archivo original paso por Excel y ~39% de los id quedaron guardados
    en notacion cientifica ('1.02E+18'). Si pandas los lee como float,
    esos miles de ids colapsan en una docena de valores identicos y
    cualquier drop_duplicates(subset='id') borra ~7,500 listings validos.
    """
    df = pd.read_csv(path, dtype={"id": "string", "host_id": "string", "license": "string"})
    log.info("crudo: %d filas x %d columnas", *df.shape)
    return df


def flag_corrupt_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca los id destruidos por Excel. No son recuperables: el redondeo
    perdio los digitos. Se conserva la fila (los demas campos estan bien)
    pero el id deja de servir como llave.
    """
    df = df.copy()
    df["id_is_corrupt"] = (
        df["id"].astype("string").str.contains("E+", regex=False, na=False).astype("int8")
    )
    n = int(df["id_is_corrupt"].sum())
    log.warning("ids corruptos por Excel: %d (%.1f%%)", n, 100 * n / len(df))
    return df


def build_surrogate_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Llave sustituta para deduplicar, ya que id no es confiable.
    host_id si sobrevivio intacto, asi que sirve de ancla.
    """
    df = df.copy()
    df["listing_key"] = (
        df["host_id"].astype("string")
        + "|" + df["latitude"].round(5).astype("string")
        + "|" + df["longitude"].round(5).astype("string")
        + "|" + df["name"].astype("string").str.slice(0, 60)
    )
    return df


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes. Sin esto, 'New ' != 'New' y el centinela se escapa."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def parse_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    rating viene como texto con DOS centinelas: 'No rating' y 'New '.
    Se separan en columnas propias porque no significan lo mismo:
    'New' es un listing sin historial todavia, 'No rating' es ausencia real.
    """
    df = df.copy()
    raw = df["rating"].astype("string")

    df["is_new_listing"] = raw.isin({"New"}).fillna(False).astype("int8")
    df["has_rating"] = (~raw.isin(RATING_SENTINELS) & raw.notna()).astype("int8")
    df["rating"] = pd.to_numeric(raw.where(~raw.isin(RATING_SENTINELS)), errors="coerce")
    return df


def parse_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    bedrooms: 'Studio' significa 0 recamaras, no un nulo. Tirarlo borra 1817 filas.
    baths: 'Not specified' si es nulo genuino (solo 13 casos).
    """
    df = df.copy()

    bed = df["bedrooms"].astype("string")
    df["is_studio"] = (bed == BEDROOMS_STUDIO).fillna(False).astype("int8")
    df["bedrooms"] = pd.to_numeric(bed.replace({BEDROOMS_STUDIO: "0"}), errors="coerce")

    bath = df["baths"].astype("string")
    df["baths"] = pd.to_numeric(bath.where(~bath.isin(BATHS_SENTINELS)), errors="coerce")

    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    return df


def parse_license(df: pd.DataFrame) -> pd.DataFrame:
    """
    license tiene inconsistencia de mayusculas: hay 'OSE-STRREG-...',
    'ose-strreg-...' y 'Ose-strreg-...'. Un startswith ingenuo pierde registros.
    Se normaliza a upper antes de clasificar.
    """
    df = df.copy()
    lic = df["license"].astype("string").str.upper()

    conditions = [
        lic.str.startswith("OSE-STRREG", na=False),
        lic == "EXEMPT",
        lic == "NO LICENSE",
    ]
    choices = ["registered", "exempt", "none"]
    df["license_status"] = pd.Categorical(
        np.select(conditions, choices, default="unknown"),
        categories=["registered", "exempt", "none", "unknown"],
    )
    df["is_registered"] = (df["license_status"] == "registered").astype("int8")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """last_review viene en formato dd/mm/yy. dayfirst evita el swap silencioso."""
    df = df.copy()
    df["last_review"] = pd.to_datetime(
        df["last_review"], format="%d/%m/%y", errors="coerce"
    )
    return df


def parse_price(df: pd.DataFrame) -> pd.DataFrame:
    """price a numerico. Solo se tira lo imposible; los outliers se tratan aparte."""
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Solo se eliminan filas sin las columnas indispensables para modelar.
    Nada de dropna() global: eso tira filas por nulos en columnas que ni usamos.
    """
    df = df.copy()
    essential = ["price", "latitude", "longitude", "room_type", "neighbourhood"]
    before = len(df)
    df = df.dropna(subset=essential)

    df = df[df["price"].between(PRICE_HARD_MIN, PRICE_HARD_MAX)]
    # Se deduplica por llave sustituta, NUNCA por id (ver flag_corrupt_ids).
    df = df.drop_duplicates(subset=["listing_key"], keep="first")
    log.info("drop_invalid: %d -> %d filas", before, len(df))
    return df


def flag_price_outliers(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """
    Marca outliers por IQR sobre log(price) en vez del corte arbitrario de $1500.
    Se MARCA, no se borra: asi el analisis decide y el corte queda documentado.
    """
    df = df.copy()
    logp = np.log(df["price"])
    q1, q3 = logp.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    df["is_price_outlier"] = (~logp.between(lo, hi)).astype("int8")
    log.info(
        "outliers de precio: %d (corte $%.0f - $%.0f)",
        df["is_price_outlier"].sum(),
        np.exp(lo),
        np.exp(hi),
    )
    return df


def clean(path) -> pd.DataFrame:
    """Pipeline completo. Este es el unico punto de entrada publico."""
    return (
        load_raw(path)
        .pipe(_strip_object_cols)
        .pipe(flag_corrupt_ids)
        .pipe(build_surrogate_key)
        .pipe(parse_price)
        .pipe(parse_rating)
        .pipe(parse_rooms)
        .pipe(parse_license)
        .pipe(parse_dates)
        .pipe(drop_invalid)
        .pipe(flag_price_outliers)
    )