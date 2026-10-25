# Contrato de entrada del preprocesamiento (v1)

Módulo único: `lsc70/preprocessing.py` → `preprocess(landmarks, is_left)`. Las matemáticas
no se reimplementan: delega en `normalizacion.vector_features` (copia textual del notebook,
protegida por prueba). Este módulo solo añade la limpieza previa y un resultado tipado.

## Entrada
- 21 landmarks de MediaPipe Hands, en coordenadas normalizadas de imagen, como objetos con
  `.x/.y/.z` o un arreglo `(21, 3)`.
- `is_left`: bandera de lateralidad de MediaPipe (`Left`). La lateralidad que se pasa es la
  que reporta MediaPipe sobre la imagen tal cual se le dio; si la app espeja la cámara
  frontal, debe corregirlo antes (no verificado aquí: vive en la app Android).

## Limpieza (rechazos, nunca excepción)
| `reason` | Condición |
|---|---|
| `bad_shape` | no son 21×3 |
| `non_finite` | hay NaN o infinito |
| `out_of_range` | x/y fuera de [-0.5, 1.5] o \|z\| > 5 (los puntos cortados por el borde, poco fuera de [0,1], se aceptan) |
| `degenerate_scale` | la norma máxima respecto a la muñeca < 1e-6 |

"No hay mano" lo decide el detector antes de llamar a `preprocess`.

## Salida
`Result(features, extras, reason)`: `features` float32 `(63,)` = muñeca al origen, división
por la **norma máxima de los 21 puntos** (no por la palma), espejo en X si es izquierda,
aplanado `[x0,y0,z0,x1,…]`; `extras` float32 `(3,)` = muñeca_x (espejada), muñeca_y, escala.

## Vectores de prueba
`tests/vectors/preprocessing_v1.json` es el contrato: 3 casos (derecha, izquierda, otra
mano) con tolerancia 1e-6. Un cambio en ese archivo es un cambio incompatible y exige subir
`CONTRACT_VERSION` y reentrenar. Se regenera con `python -m tests.make_vectors`.
`tests/test_preprocessing.py` además compara contra una reimplementación independiente y
contra la copia del notebook.
