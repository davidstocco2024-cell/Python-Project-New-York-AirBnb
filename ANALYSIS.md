# ANALYSIS.md — segunda pasada sobre el dataset

`README.md` es el documento original del repo y se deja tal cual. Este archivo
documenta lo que se agrego encima: auditoria del CSV, pipeline reproducible y
modelo con validacion espacial.

`note.ipynb` tambien queda intacto. El notebook nuevo es `analysis.ipynb` y es
independiente: no comparte estado con el original ni hace falta correr uno
para que funcione el otro.

---

## Bugs del dataset

Los cuatro salieron auditando `datasets.csv` y cada uno tiene su test de
regresion en `tests/test_cleaning.py`.

**1. Los `id` pasaron por Excel.** 8,050 filas (38.8%) traen el id guardado
como `1.02E+18`. No es como pandas lo muestra: el texto en el archivo dice
literalmente eso. Si se lee como float, esos miles de ids colapsan en una
docena de valores y un `drop_duplicates(subset="id")` borra ~7,500 listings
validos. Se leen como texto, se marcan con `id_is_corrupt` y se deduplica con
una llave sustituta de `host_id` + coordenadas + nombre. `host_id` sobrevivio
intacto.

**2. `rating`, `baths` y `bedrooms` son texto, no numeros.** Traen centinelas
mezclados: `'No rating'` (3,595), `'New '` con espacio al final (159),
`'Not specified'` (13) y `'Studio'` (1,817). Sin castear, esas tres columnas
desaparecen en silencio de cualquier `.corr()` o `.describe()`. Resultan ser
la segunda y tercera feature mas importantes del modelo.

**3. `license` tiene mayusculas inconsistentes.** Hay `OSE-STRREG-...`,
`ose-strreg-...` y `Ose-strreg-...`. Un `startswith` ingenuo pierde registros.

**4. El corte de precio en $1,500 es arbitrario.** Se sustituye por IQR sobre
`log(price)`, que da un corte de ~$3,063 y marca 25 outliers en vez de
borrarlos.

---

## Hallazgo principal: Local Law 18

El snapshot es de enero de 2024, cuatro meses despues de que NYC empezara a
aplicar la Local Law 18, que obliga a registrar los alquileres de corta
estancia. La ley solo regula estancias **menores a 30 noches**.

| estatus de licencia | listings | usa el minimo de 30 noches | reviews ultimos 12m (mediana) |
|---|---|---|---|
| registered | 1,048 | 4.3% | 26 |
| exempt | 2,109 | 2.8% | 6 |
| none | 17,426 | **99.96%** | 3 |

De los 17,426 listings sin licencia, 17,419 exigen 30 noches o mas. Quedan
siete excepciones en todo el dataset: practicamente no hay nadie operando
corta estancia sin registro. Los que si se registraron reciben alrededor de
8 veces mas reservas.

La distribucion bimodal de `availability_365` que el EDA original lee como
"estrategia del host" se explica mejor por esta bifurcacion regulatoria.

`license_status` y `uses_30night_loophole` terminan siendo casi la misma
variable, asi que para el modelo son colineales y conviene usar solo una.

---

## Resultados del modelo

Regresion de `log(price)` con LightGBM. Se reportan dos esquemas de validacion
porque la diferencia entre ambos es informativa: con split aleatorio el
departamento de enfrente cae en train y el modelo memoriza la cuadra en vez de
aprender.

| features | R2 aleatorio | R2 por barrio | MAE |
|---|---|---|---|
| A) las del notebook original | 0.618 | 0.526 | $69 |
| B) + columnas rescatadas de la limpieza | 0.671 | 0.607 | $61 |
| C) + distancias geo y variables LL18 | 0.684 | **0.618** | $60 |

Arreglar la limpieza (A → B) subio el R2 espacial 0.081. Las features
geograficas (B → C) subieron 0.011. El bug fix valio ocho veces mas que la
ingenieria de features, que es lo normal y casi nunca se reporta.

**Leakage:** el notebook original crea `price per bed = price / beds`. Con esa
columna dentro, el modelo saca R2 de 0.99 y no sirve para nada.
`modeling.assert_no_leakage` falla ruidosamente si alguna feature contiene
"price".

---

## Lo que este dataset NO permite

Es una foto estatica. `last_review` es la unica fecha y no forma una serie de
tiempo, asi que **no hay forecasting posible** — ni ARIMA, ni Prophet, ni
prediccion de ocupacion futura.

Ademas, los 8,050 ids dañados no son una muestra aleatoria: son listings mas
recientes y difieren sistematicamente (78.6% vs 89.2% en el uso del minimo de
30 noches). Cualquier analisis a nivel listing individual carga ese sesgo.

---

## Datos geograficos sin pagar Google

Google retiro el credito mensual de $200 en marzo de 2025 y lo cambio por
topes gratuitos por SKU. Nearby Search en tier Pro cuesta $32 USD por cada
1,000 llamadas, asi que enriquecer los 20,600 listings una sola vez costaria
alrededor de $650.

`scripts/fetch_geo_data.py` baja lo mismo gratis de Overpass (OpenStreetMap):
entradas de metro y POIs de los cinco boroughs. Las distancias se calculan con
`BallTree` en metrica haversine, 20k listings contra cientos de estaciones en
menos de un segundo.

Google Maps tiene sentido en la capa de producto, no en el ETL: cuando el
usuario hace clic en un listing y quiere tiempo real en transporte publico.
Son unas pocas llamadas por sesion en vez de 20,600.

---

## Como correr

```bash
pip install -r requirements.txt

python scripts/01_clean.py          # OBLIGATORIO: datasets.csv -> data/processed/
python scripts/fetch_geo_data.py    # opcional, requiere internet
python scripts/03_train.py          # depende del 01

pytest tests/ -q                    # independiente, usa datos sinteticos
```

Solo `01_clean.py` es prerequisito de `03_train.py`. Si no corres
`fetch_geo_data.py`, el pipeline detecta que faltan los CSVs, avisa y sigue
sin esas features.

Las features no tienen script propio: se construyen en memoria dentro de
`03_train.py` y de `analysis.ipynb` con `build_features()`. Son baratas y no
ameritan un parquet intermedio, pero implica que la unica fuente de verdad es
`src/features.py`.

---

## Que se agrego al repo

```
src/config.py               rutas y constantes (datasets.csv sigue en la raiz)
src/cleaning.py             limpieza, una funcion por transformacion
src/features.py             haversine, BallTree, features de LL18 y de host
src/modeling.py             GroupKFold, guarda de leakage, importancias
scripts/01_clean.py         ejecuta la limpieza y persiste
scripts/fetch_geo_data.py   descarga Overpass (una sola vez)
scripts/03_train.py         entrena y compara los tres sets de features
tests/test_cleaning.py      un test por cada bug encontrado
analysis.ipynb              notebook nuevo, independiente de note.ipynb
requirements.txt
```

Sin cambios: `README.md`, `note.ipynb`, `datasets.csv`, `LICENSE`, `images/`
y el `.docx`.
