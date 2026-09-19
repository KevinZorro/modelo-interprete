"""Carga de datos y reproducción exacta del split con el que se entrenó el modelo.

Por qué esto es necesario: el `.tflite` desplegado se entrenó sobre `X_R[i_tr]`, un
split por participante hecho en el notebook con `GroupShuffleSplit(seed=42)` sobre
las letras MÁS 800 negativos de HaGRID agrupados individualmente. Para medir algo
honesto hay que evaluar SOLO sobre los participantes que ese modelo nunca vio, y
para saber quiénes son hay que reconstruir el mismo split, con el mismo orden de
grupos y la misma semilla.

La reconstrucción se verifica contra las cifras que quedaron impresas en el
notebook (train 8992 / test 2617 / 171 negativos). Si algún día cambia el cache o
la semilla, esas asserts fallan en vez de devolver números silenciosamente malos.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# --- Constantes del split original (celda 2 y 68 del notebook v8) -------------
SEED = 42
N_PARTICIPANTES_TEST = 15
N_NEGATIVOS = 800          # negativos de HaGRID usados al entrenar el rechazo
IDX_RECHAZO = 27           # columna 28 de la salida: "no_es_seña"

# Cifras esperadas de la corrida original. Son la prueba de que el split
# reconstruido es el mismo, no uno parecido.
ESPERADO = {"train": 8992, "test": 2617, "negativos_test": 171, "tomas_test": 428}


def cargar_cache(ruta_cache):
    """Devuelve (X, y, participante, nombre, etiquetas) con los dígitos ya fuera.

    El cache guarda train/test por separado; aquí se concatenan porque el split
    que importa es el de la sección de rechazo, que se rehízo sobre el total.
    """
    d = np.load(ruta_cache, allow_pickle=True)
    etiquetas_crudas = [str(e) for e in d["etiquetas"]]

    # 'none' son los dígitos: se copiaron a disco pero nunca se entrenaron.
    i_none = etiquetas_crudas.index("none")
    etiquetas = [e for e in etiquetas_crudas if e != "none"]
    remap = {etiquetas_crudas.index(e): i for i, e in enumerate(etiquetas)}

    def _filtrar(sufijo):
        m = d[f"y{sufijo}"] != i_none
        return (d[f"X{sufijo}"][m],
                np.array([remap[v] for v in d[f"y{sufijo}"][m]]),
                d[f"p{sufijo}"][m].astype(str),
                d[f"n{sufijo}"][m].astype(str))

    partes = [_filtrar("tr"), _filtrar("te")]
    X, y, part, nom = (np.concatenate([a[i] for a in partes]) for i in range(4))

    # Integridad: el nombre de archivo lleva la clase real ('Per01__A__...').
    malos = sum(1 for nm, yy in zip(nom, y) if nm.split("__")[1] != etiquetas[yy])
    assert malos == 0, f"{malos} nombres desalineados con su etiqueta"
    return X, y, part, nom, etiquetas


def indices_held_out(y, participantes):
    """Índices de las muestras de letra que quedaron FUERA del entrenamiento.

    Reconstruye el split de la celda 68: los grupos son los participantes para
    las letras y un grupo distinto por cada negativo de HaGRID (así se hizo allí
    porque el dataset no trae user_id). El orden importa: GroupShuffleSplit
    permuta `np.unique(grupos)`, que es el conjunto ordenado alfabéticamente.
    """
    grupos_neg = np.array([f"hagrid_neg_idx_{i}" for i in range(N_NEGATIVOS)])
    grupos = np.concatenate([participantes, grupos_neg])
    y_r = np.concatenate([y, np.full(N_NEGATIVOS, IDX_RECHAZO)])

    frac_test = N_PARTICIPANTES_TEST / len(set(participantes))
    gss = GroupShuffleSplit(n_splits=1, test_size=frac_test, random_state=SEED)
    i_tr, i_te = next(gss.split(np.zeros(len(y_r)), y_r, groups=grupos))

    assert not (set(grupos[i_tr]) & set(grupos[i_te])), "FUGA DE PARTICIPANTE"
    assert len(i_tr) == ESPERADO["train"], f"train {len(i_tr)} != {ESPERADO['train']}"
    assert len(i_te) == ESPERADO["test"], f"test {len(i_te)} != {ESPERADO['test']}"
    assert (i_te >= len(y)).sum() == ESPERADO["negativos_test"]

    # Los negativos no están en el cache (solo letras), así que se descartan aquí.
    return i_te[i_te < len(y)]


def probabilidades(ruta_modelo, X):
    """Pasa X por el .tflite desplegado. Devuelve (n, 28) de probabilidades.

    No se reentrena nada: es exactamente el modelo que está en la app.
    """
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(ruta_modelo))
    entrada = it.get_input_details()[0]
    it.resize_tensor_input(entrada["index"], [len(X), X.shape[1]])
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], X.astype(np.float32))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def agrupar_por_toma(nombres):
    """'Per01__A__Per01_A_3.jpg' -> toma 'Per01__A'.

    Cada participante grabó cada seña una sola vez, así que participante+clase
    identifica la toma sin necesidad de un regex frágil sobre el nombre.
    """
    tomas = defaultdict(list)
    for j, nm in enumerate(nombres):
        tomas["__".join(str(nm).split("__")[:2])].append(j)
    return tomas


def datos_evaluacion(ruta_cache, ruta_modelo):
    """Punto de entrada único: probabilidades por frame y por toma del held-out.

    Devuelve un dict con todo lo que necesitan la calibración y las métricas.
    La agregación por toma es el PROMEDIO de las probabilidades de sus frames,
    que es como se midió el 88.8 % y lo que hace la app en ~medio segundo.
    """
    X, y, part, nom, etiquetas = cargar_cache(ruta_cache)
    i_te = indices_held_out(y, part)
    P_frame = probabilidades(ruta_modelo, X[i_te])

    y_te, part_te, nom_te = y[i_te], part[i_te], nom[i_te]
    tomas = agrupar_por_toma(nom_te)
    claves = sorted(tomas)
    assert len(claves) == ESPERADO["tomas_test"], f"{len(claves)} tomas"

    return {
        "etiquetas": etiquetas,
        "P_toma": np.stack([P_frame[tomas[k]].mean(0) for k in claves]),
        "P_frames_por_toma": [P_frame[tomas[k]] for k in claves],
        "y_toma": np.array([y_te[tomas[k][0]] for k in claves]),
        "part_toma": np.array([part_te[tomas[k][0]] for k in claves]),
        "claves": claves,
    }


def cargar_negativos(ruta, ruta_modelo, clases_excluidas=()):
    """Landmarks de 'mano en reposo' exportados desde Colab (opcional).

    Espera un .npz con la matriz (n, 63) bajo la clave 'X_neg' o la primera que
    haya; lo produce la celda '15-bis. Exportar los negativos de reposo' del
    notebook. Sin este archivo no se puede medir la falsa aceptación de reposo, y el
    config lo deja explícitamente como pendiente en vez de heredar una cifra
    vieja: los umbrales cambian, así que la cifra vieja ya no aplica.
    """
    if ruta is None or not Path(ruta).exists():
        return None
    d = np.load(ruta, allow_pickle=True)
    clave = "X_neg" if "X_neg" in d.files else d.files[0]
    X_neg = d[clave]
    assert X_neg.shape[1] == 63, f"esperaba (n, 63), llegó {X_neg.shape}"

    # Las "letras disfrazadas" no son negativos: son la misma pose que una letra
    # con otro nombre. Dejarlas dentro infla el umbral de esa letra.
    if clases_excluidas:
        if "clase_por_muestra" not in d.files:
            raise SystemExit(
                "El .npz no trae 'clase_por_muestra', así que no se pueden filtrar "
                f"{list(clases_excluidas)}. Reexporta con la celda 15-bis del notebook."
            )
        clases = d["clase_por_muestra"].astype(str)
        m = ~np.isin(clases, list(clases_excluidas))
        print(f"Negativos: {len(X_neg)} -> {int(m.sum())} tras excluir "
              f"{list(clases_excluidas)} (letras disfrazadas)")
        X_neg = X_neg[m]
    return probabilidades(ruta_modelo, X_neg)
