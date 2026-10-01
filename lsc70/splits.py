"""Splits por participante, UNA sola fuente de verdad para ANH y AN.

Por qué importa: ANH y AN son las MISMAS 70 personas y las MISMAS tomas, vistas de dos
maneras. Si cada sección se partiera por su cuenta, una misma persona podría quedar en
train en una y en test en la otra, y al combinar secciones habría fuga. Aquí se decide
una vez qué participante va a dónde y todo lo demás lo consulta.

Tres niveles, siempre por participante (nunca por frame ni por toma):
  1. test   : 15 participantes, la MISMA regla que el notebook (semilla 42), para que los
              resultados se puedan comparar con los anteriores.
  2. folds  : 5 grupos de participantes para validación cruzada (cada persona se evalúa
              una vez con un modelo que no la vio). Es la evaluación principal: n = 70
              personas por clase, no 15.
  3. val    : dentro de cada entrenamiento, 15 % de los participantes de train apartados
              para early stopping (lo hace entrenar_evaluar, igual que `entrenar` del
              notebook).
"""

import json
from pathlib import Path

import numpy as np

SEED = 42
N_TEST = 15
N_FOLDS = 5
RUTA = Path(__file__).resolve().parent.parent / "configs" / "splits_participantes.json"


def construir(participantes, seed=SEED, n_test=N_TEST, n_folds=N_FOLDS):
    parts = sorted(set(participantes))
    # Regla de la celda 10 del notebook, tal cual.
    perm = np.random.default_rng(seed).permutation(len(parts))
    test = sorted(parts[i] for i in perm[:n_test])
    train = sorted(parts[i] for i in perm[n_test:])
    # Folds: otra permutación (seed + 1) para que no dependan de la del test.
    orden = np.random.default_rng(seed + 1).permutation(len(parts))
    fold = {parts[i]: int(r % n_folds) for r, i in enumerate(orden)}
    assert not set(test) & set(train)
    return {
        "seed": seed,
        "n_participantes": len(parts),
        "test": test,
        "train": train,
        "fold_de": fold,
    }


def guardar(splits, ruta=RUTA):
    ruta.parent.mkdir(exist_ok=True)
    ruta.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar(participantes, ruta=RUTA):
    """Carga los splits; si no existen, los crea. Falla si los participantes no coinciden:
    significa que otra sección tiene gente distinta y los splits ya no se pueden compartir."""
    if ruta.exists():
        s = json.loads(ruta.read_text(encoding="utf-8"))
    else:
        s = construir(participantes)
        guardar(s, ruta)
    faltan = set(participantes) - set(s["fold_de"])
    if faltan:
        raise SystemExit(
            f"Participantes sin split asignado: {sorted(faltan)[:5]}. "
            "¿Otra sección con gente distinta? Revisa antes de seguir."
        )
    return s
