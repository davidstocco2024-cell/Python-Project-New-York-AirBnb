import numpy as np
import pytest

from src.config import MODEL_FILE
from src.predict import Bundle, _align, build_input_row, explain, predict_price

pytestmark = pytest.mark.skipif(
    not MODEL_FILE.exists(), reason="corre scripts/04_export_model.py primero"
)


@pytest.fixture(scope="module")
def bundle():
    return Bundle.load(MODEL_FILE)


def _row(**kw):
    base = dict(
        latitude=40.7580,
        longitude=-73.9855,
        neighbourhood_group="Manhattan",
        neighbourhood="Midtown",
        room_type="Entire home/apt",
        bedrooms=1,
        beds=1,
        baths=1,
        minimum_nights=30,
        license_status="none",
        listings_in_neighbourhood=400,
    )
    return build_input_row(**{**base, **kw})


def test_precio_en_rango_razonable(bundle):
    assert 20 < predict_price(bundle, _row()) < 5000


def test_la_app_arma_todas_las_features_del_modelo(bundle):
    """Si build_input_row se desincroniza de features.py, esto lo caza."""
    faltantes = set(bundle.feature_cols) - set(_row().columns)
    assert not faltantes, f"el modelo espera features que la app no arma: {faltantes}"


def test_manhattan_cuesta_mas_que_el_bronx(bundle):
    manhattan = predict_price(bundle, _row())
    bronx = predict_price(
        bundle,
        _row(
            latitude=40.8448,
            longitude=-73.8648,
            neighbourhood_group="Bronx",
            neighbourhood="Fordham",
            listings_in_neighbourhood=50,
        ),
    )
    assert manhattan > bronx


def test_cuarto_privado_cuesta_menos_que_departamento_entero(bundle):
    assert predict_price(bundle, _row(room_type="Private room")) < predict_price(bundle, _row())


def test_las_contribuciones_suman_a_la_prediccion(bundle):
    X = _row()
    contrib = bundle.model.predict(_align(X, bundle), pred_contrib=True)[0]
    assert np.isclose(contrib.sum(), np.log(predict_price(bundle, X)), atol=1e-6)


def test_explain_devuelve_contribuciones_no_triviales(bundle):
    e = explain(bundle, _row(), top=5)
    assert len(e) == 5
    assert e["efecto_pct"].abs().sum() > 0


def test_las_categorias_se_alinean_por_nombre_no_por_posicion(bundle):
    """
    LightGBM codifica categoricas por posicion. Si el orden cambiara,
    'Manhattan' podria volverse 'Queens' sin que nada truene.
    """
    X = _align(_row(neighbourhood_group="Queens"), bundle)
    cats = list(X["neighbourhood_group"].cat.categories)
    assert cats == bundle.categories["neighbourhood_group"]
    assert X["neighbourhood_group"].iloc[0] == "Queens"