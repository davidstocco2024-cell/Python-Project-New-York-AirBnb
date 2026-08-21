import pandas as pd
import pytest

from src.cleaning import (
    _strip_object_cols,
    build_surrogate_key,
    flag_corrupt_ids,
    parse_license,
    parse_rating,
    parse_rooms,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "id": ["1312228", "1.02E+18", "45277537"],
            "host_id": ["7130382", "544652279", "51501835"],
            "name": ["Loft A", "Home B", "Unit C"],
            "latitude": [40.68371, 40.81, 40.76661],
            "longitude": [-73.96461, -73.94, -73.9881],
            # 'New ' con espacio al final, tal cual viene en el archivo
            "rating": ["5", "New ", "No rating"],
            "bedrooms": ["1", "Studio", "2"],
            "baths": ["1", "Not specified", "1.5"],
            "beds": [1, 2, 1],
            "license": ["No License", "ose-strreg-0001021", "Exempt"],
        }
    )


def test_rating_detecta_ambos_centinelas(raw):
    out = parse_rating(_strip_object_cols(raw))
    # 'New ' debe sobrevivir al strip y clasificarse, no volverse un rating numerico
    assert out["rating"].isna().sum() == 2
    assert out["rating"].iloc[0] == 5.0
    assert out["is_new_listing"].tolist() == [0, 1, 0]
    assert out["has_rating"].tolist() == [1, 0, 0]


def test_studio_es_cero_no_nulo(raw):
    out = parse_rooms(_strip_object_cols(raw))
    assert out["bedrooms"].tolist() == [1.0, 0.0, 2.0]
    assert out["is_studio"].tolist() == [0, 1, 0]
    assert out["bedrooms"].isna().sum() == 0  # no se pierde ninguna fila


def test_license_ignora_mayusculas(raw):
    out = parse_license(_strip_object_cols(raw))
    # el registro en minusculas debe contar como 'registered'
    assert out["license_status"].tolist() == ["none", "registered", "exempt"]
    assert out["is_registered"].sum() == 1


def test_id_cientifico_se_marca(raw):
    out = flag_corrupt_ids(_strip_object_cols(raw))
    assert out["id_is_corrupt"].tolist() == [0, 1, 0]


def test_llave_sustituta_es_unica(raw):
    out = build_surrogate_key(_strip_object_cols(raw))
    assert out["listing_key"].nunique() == len(out)


def test_ids_corruptos_no_colapsan_al_leer_como_texto():
    ids = pd.Series(["1.02E+18", "1.02E+18", "1312228"])
    como_float = pd.to_numeric(ids, errors="coerce")
    assert como_float.nunique() == 2  # dos ids distintos colapsan en uno
    assert ids.nunique() == 2  # como texto se conserva la distincion original