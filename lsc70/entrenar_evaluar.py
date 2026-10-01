"""Tareas 5 y 6: entrenar con letras + números y evaluar top-1 por toma.

    python -m lsc70.entrenar_evaluar --exp anh_letras_numeros \\
        --npz anh=landmarks_anh.npz --entrenar-con anh --evaluar-en anh \\
        --clases letras+numeros --semillas 0 42 123

Protocolo (el mismo para todas las secciones, para que las cifras se puedan comparar):
  - 5 folds por PARTICIPANTE (configs/splits_participantes.json): cada persona se evalúa una
    vez con modelos que nunca la vieron. 70 personas por clase, no 15.
  - Dentro de cada entrenamiento, 15 % de los participantes de train se apartan como
    validación para el early stopping. Tres niveles, siempre por persona.
  - La unidad de evaluación es la TOMA (participante, clase): se promedian las
    probabilidades de sus ~6 frames, como en el 88.8 % de referencia.
  - Red e hiperparámetros: los del notebook v8, sin tocar. Varias semillas.

Tres lecturas del top-1, porque responden preguntas distintas:
  estricto        : acierto sobre las tomas donde MediaPipe vio la mano. Comparable con 88.8 %.
  extremo a extremo: las tomas sin ninguna detección cuentan como fallo. Es lo que vive la
                    persona frente a una cámara siempre encendida.
  con equivalencias: acierta si la predicha es la MISMA pose que la real con otro nombre
                    (p. ej. V y el 2). No es un acierto "regalado": es lo que ningún modelo
                    de poses puede separar. Se descubren empíricamente, no se inventan.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from evaluacion_objetivo.analisis_equivalencias import MAX_ASIMETRIA, MIN_COACTIVACION
from evaluacion_objetivo.metricas import wilson

from . import splits as mod_splits

LETRAS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "NN",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
NUMEROS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
DINAMICAS_CONOCIDAS = ["Z", "NN"]  # el modelo es de poses estáticas

# Hiperparámetros: los de la celda de configuración del notebook v8.
BATCH_SIZE, LEARNING_RATE, EPOCHS_MAX, PACIENCIA, FRAC_VAL = 16, 0.001, 200, 20, 0.15


# --- Modelo (copia de las celdas 'Utilidades de entrenamiento' del notebook) --------
def construir_red(n_features, n_clases, unidades=(256, 128), dropout=(0.4, 0.3)):
    from tensorflow import keras

    capas = [keras.layers.Input(shape=(n_features,)), keras.layers.BatchNormalization()]
    for u, d in zip(unidades, dropout):
        capas += [keras.layers.Dense(u, activation="relu"), keras.layers.Dropout(d)]
    capas.append(keras.layers.Dense(n_clases, activation="softmax"))
    m = keras.Sequential(capas)
    m.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def entrenar(X, y, grupos, n_clases, seed, epochs=EPOCHS_MAX):
    """Validación separada POR PARTICIPANTE; nunca validation_split de Keras (toma el
    último trozo del array sin barajar y deja clases enteras fuera)."""
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    i_tr, i_val = next(
        GroupShuffleSplit(1, test_size=FRAC_VAL, random_state=seed).split(
            np.zeros(len(y)), y, groups=grupos
        )
    )
    assert not set(grupos[i_tr]) & set(grupos[i_val]), "Fuga train/val"
    m = construir_red(X.shape[1], n_clases)
    m.fit(
        X[i_tr],
        y[i_tr],
        validation_data=(X[i_val], y[i_val]),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PACIENCIA, restore_best_weights=True
            )
        ],
    )
    return m


# --- Datos ------------------------------------------------------------------------
def elegir_clases(modo, presentes, incluir_palabras=False, lista=None):
    """Orden fijo y reproducible: letras primero, luego números."""
    if lista:
        sel = lista
    else:
        sel = [c for c in LETRAS if c in presentes]
        if modo == "letras+numeros":
            sel += [c for c in NUMEROS if c in presentes]
            if incluir_palabras:
                sel += [c for c in ("MIL", "MILLON") if c in presentes]
    faltan = [c for c in sel if c not in presentes]
    if faltan:
        raise SystemExit(f"Clases pedidas que no existen en los datos: {faltan}")
    return sel


def cargar(npz):
    d = np.load(npz, allow_pickle=True)
    return {k: d[k] for k in d.files}


def a_tomas(P_frames, y, part, secciones):
    """Promedia las probabilidades de los frames de cada toma (por sección)."""
    claves = {}
    for i, (s, p, c) in enumerate(zip(secciones, part, y)):
        claves.setdefault((s, p, int(c)), []).append(i)
    orden = sorted(claves)
    P = np.stack([P_frames[claves[k]].mean(0) for k in orden])
    return P, orden


def grupos_equivalencia(P, y, K):
    """Pares de clases que el modelo no puede separar, medidos con la co-activación:
    probabilidad media de la clase j cuando la real es i, fuerte en AMBAS direcciones y
    parecida entre ellas. Mismos umbrales que evaluacion_objetivo.analisis_equivalencias."""
    M = np.zeros((K, K))
    for i in range(K):
        if (y == i).sum() >= 5:
            M[i] = P[y == i].mean(0)
    padre = list(range(K))

    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    pares = []
    for i in range(K):
        for j in range(i + 1, K):
            lo, hi = min(M[i, j], M[j, i]), max(M[i, j], M[j, i])
            if lo >= MIN_COACTIVACION and hi / max(lo, 1e-9) <= MAX_ASIMETRIA:
                padre[raiz(i)] = raiz(j)
                pares.append((i, j, float(M[i, j]), float(M[j, i])))
    return np.array([raiz(i) for i in range(K)]), pares


# --- Experimento ------------------------------------------------------------------
def correr(args):
    secciones = dict(kv.split("=") for kv in args.npz)
    datos = {s: cargar(r) for s, r in secciones.items()}
    presentes = set().union(*[set(d["y"].astype(str)) for d in datos.values()])
    clases = elegir_clases(
        args.clases,
        presentes,
        args.incluir_palabras,
        args.clases_lista.split(",") if args.clases_lista else None,
    )
    K = len(clases)
    idx = {c: i for i, c in enumerate(clases)}
    print(f"Clases ({K}): {clases}")

    # Todas las secciones deben tener la MISMA gente: si no, los splits no se comparten.
    todos_part = sorted(
        set().union(*[set(d["participante"].astype(str)) for d in datos.values()])
    )
    sp = mod_splits.cargar(todos_part)
    fold_de = sp["fold_de"]

    def preparar(s):
        d = datos[s]
        m = np.isin(d["y"].astype(str), clases)
        y = np.array([idx[c] for c in d["y"].astype(str)[m]])
        return d["X"][m], y, d["participante"].astype(str)[m], m

    prep = {s: preparar(s) for s in datos}
    sec_entrenar, sec_evaluar = args.entrenar_con, args.evaluar_en
    semillas = args.semillas

    # Listas de evaluación (todas las muestras de las secciones de evaluación).
    Xe = np.concatenate([prep[s][0] for s in sec_evaluar])
    ye = np.concatenate([prep[s][1] for s in sec_evaluar])
    pe = np.concatenate([prep[s][2] for s in sec_evaluar])
    se = np.concatenate([[s] * len(prep[s][1]) for s in sec_evaluar])
    fe = np.array([fold_de[p] for p in pe])

    P_sem = []
    for seed in semillas:
        P = np.zeros((len(ye), K), np.float32)
        for fold in range(mod_splits.N_FOLDS):
            Xt = np.concatenate([prep[s][0] for s in sec_entrenar])
            yt = np.concatenate([prep[s][1] for s in sec_entrenar])
            pt = np.concatenate([prep[s][2] for s in sec_entrenar])
            m_tr = np.array([fold_de[p] != fold for p in pt])
            m_te = fe == fold
            # Las personas del fold de evaluación NO pueden estar en entrenamiento.
            assert not set(pt[m_tr]) & set(pe[m_te]), "FUGA DE PARTICIPANTE"
            modelo = entrenar(Xt[m_tr], yt[m_tr], pt[m_tr], K, seed)
            P[m_te] = modelo.predict(Xe[m_te], verbose=0)
            print(
                f"  semilla {seed} fold {fold + 1}/{mod_splits.N_FOLDS} listo",
                flush=True,
            )
        P_sem.append(P)

    # --- A nivel de toma ---
    Pt_sem, orden = zip(*[a_tomas(P, ye, pe, se) for P in P_sem])
    orden = orden[0]
    y_t = np.array([k[2] for k in orden])
    sec_t = np.array([k[0] for k in orden])
    part_t = np.array([k[1] for k in orden])
    Pt = np.mean(Pt_sem, axis=0)

    perdidas = 0
    for s in sec_evaluar:
        for t in datos[s].get("tomas_perdidas", np.array([])).astype(str):
            if t.split("__")[1] in idx:
                perdidas += 1

    def top1(Pm, grupo=None):
        pred = Pm.argmax(1)
        ok = (pred == y_t) if grupo is None else (grupo[pred] == grupo[y_t])
        return float(ok.mean())

    estricto = [top1(P) for P in Pt_sem]
    print(
        f"\n=== TOP-1 POR TOMA ({len(y_t)} tomas, {len(set(part_t))} participantes) ==="
    )
    print(
        f"estricto           : {100 * np.mean(estricto):.1f} % ± {100 * np.std(estricto):.1f} "
        f"(semillas {semillas}) | ensamble {100 * top1(Pt):.1f} %"
    )
    e2e = (top1(Pt) * len(y_t)) / (len(y_t) + perdidas)
    print(
        f"extremo a extremo  : {100 * e2e:.1f} %  ({perdidas} tomas sin ninguna detección cuentan como fallo)"
    )

    grupo, pares = grupos_equivalencia(Pt, y_t, K)
    print(
        f"con equivalencias  : {100 * top1(Pt, grupo):.1f} %  "
        f"({len(pares)} pares: {[(clases[i], clases[j]) for i, j, _, _ in pares] or 'ninguno'})"
    )

    es_let = np.array([clases[c] in LETRAS for c in y_t])
    es_num = ~es_let
    pred = Pt.argmax(1)
    sub = {}
    if es_let.any():
        sub["letras"] = float((pred[es_let] == y_t[es_let]).mean())
        print(
            f"solo tomas de letras  : {100 * sub['letras']:.1f} %  (n={es_let.sum()})"
        )
        if es_num.any():
            let_a_num = np.mean([clases[p] in NUMEROS for p in pred[es_let]])
            print(f"   de ellas, predichas como número: {100 * let_a_num:.1f} %")
    if es_num.any():
        sub["numeros"] = float((pred[es_num] == y_t[es_num]).mean())
        print(
            f"solo tomas de números : {100 * sub['numeros']:.1f} %  (n={es_num.sum()})"
        )
    estat = np.array([clases[c] not in DINAMICAS_CONOCIDAS for c in y_t])
    print(
        f"sin Z ni Ñ (estáticas): {100 * float((pred[estat] == y_t[estat]).mean()):.1f} %"
    )

    # --- Por clase, matriz de confusión completa ---
    C = np.zeros((K, K), int)
    for a, b in zip(y_t, pred):
        C[a, b] += 1
    print(
        f"\n{'clase':>6} {'n':>4} {'recall':>7} {'IC95':>14}  principales confusiones"
    )
    por_clase = {}
    for i, c in enumerate(clases):
        n = int(C[i].sum())
        if not n:
            continue
        p, lo, hi = wilson(int(C[i, i]), n)
        err = sorted(
            ((C[i, j], clases[j]) for j in range(K) if j != i and C[i, j]), reverse=True
        )[:3]
        por_clase[c] = {"n": n, "recall": p, "ic95": [lo, hi]}
        print(
            f"{c:>6} {n:>4} {p:>7.3f} [{lo:.2f}-{hi:.2f}]  "
            + ", ".join(f"{n}x{c2}" for n, c2 in err)
        )

    print(
        "\n=== LETRAS BAJO SOSPECHA (C, K, M; y E, H, que son las flojas en dataset) ==="
    )
    for c in ("C", "K", "M", "E", "H"):
        if c in por_clase:
            r = por_clase[c]
            print(
                f"  {c}: recall {r['recall']:.3f} [{r['ic95'][0]:.2f}-{r['ic95'][1]:.2f}] (n={r['n']})"
            )

    # --- Guardar con todo lo necesario para reproducir ---
    salida = Path(args.salida) / args.exp
    salida.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        salida / "confusion.csv",
        C,
        fmt="%d",
        delimiter=",",
        header=",".join(clases),
        comments="",
    )
    np.savez_compressed(
        salida / "predicciones.npz",
        P=Pt,
        y=y_t,
        seccion=sec_t,
        participante=part_t,
        etiquetas=np.array(clases),
        clave=np.array([f"{p}__{clases[c]}" for p, c in zip(part_t, y_t)]),
    )

    def sha(r):
        return hashlib.sha256(Path(r).read_bytes()).hexdigest()[:16]

    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git = "desconocido"
    import tensorflow as tf

    (salida / "metricas.json").write_text(
        json.dumps(
            {
                "experimento": args.exp,
                "clases": clases,
                "entrenar_con": sec_entrenar,
                "evaluar_en": sec_evaluar,
                "semillas": semillas,
                "n_tomas": len(y_t),
                "n_participantes": len(set(part_t)),
                "tomas_perdidas": perdidas,
                "top1_estricto_por_semilla": estricto,
                "top1_estricto_ensamble": top1(Pt),
                "top1_extremo_a_extremo": e2e,
                "top1_con_equivalencias": top1(Pt, grupo=grupo),
                "pares_equivalentes": [
                    (clases[i], clases[j], a, b) for i, j, a, b in pares
                ],
                "subconjuntos": sub,
                "por_clase": por_clase,
                "reproducibilidad": {
                    "git": git,
                    "tensorflow": tf.__version__,
                    "datos_sha256": {s: sha(r) for s, r in secciones.items()},
                    "hiperparametros": {
                        "batch": BATCH_SIZE,
                        "lr": LEARNING_RATE,
                        "epochs_max": EPOCHS_MAX,
                        "paciencia": PACIENCIA,
                        "val_frac": FRAC_VAL,
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nEscrito en {salida}/ (metricas.json, confusion.csv, predicciones.npz)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--exp", required=True)
    ap.add_argument(
        "--npz", nargs="+", required=True, help="seccion=ruta.npz (uno o varios)"
    )
    ap.add_argument("--entrenar-con", nargs="+", required=True)
    ap.add_argument("--evaluar-en", nargs="+", required=True)
    ap.add_argument(
        "--clases", choices=["letras", "letras+numeros"], default="letras+numeros"
    )
    ap.add_argument(
        "--clases-lista", default=None, help="lista explícita separada por comas"
    )
    ap.add_argument(
        "--incluir-palabras", action="store_true", help="añade MIL y MILLON"
    )
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--salida", default="resultados")
    correr(ap.parse_args())


if __name__ == "__main__":
    main()
