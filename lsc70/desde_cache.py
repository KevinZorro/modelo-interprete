"""Convierte landmarks_cache.npz (formato del notebook) al formato de lsc70.extraer.

Solo conserva las 27 letras: la clase 'none' del cache mezcla los números en un
submuestreo de 400 por split, tomados por orden de participante (sesgado), así que no
sirve para entrenar números por separado. Para eso hay que reextraer ANH con extraer.py.

    python -m lsc70.desde_cache --cache landmarks_cache.npz --salida landmarks_anh_letras.npz
"""

import argparse

import numpy as np


def convertir(ruta_cache, salida):
    d = np.load(ruta_cache, allow_pickle=True)
    et = [str(e) for e in d["etiquetas"]]
    X = np.concatenate([d["Xtr"], d["Xte"]])
    E = np.concatenate([d["Etr"], d["Ete"]])
    y = np.array([et[i] for i in np.concatenate([d["ytr"], d["yte"]])])
    part = np.concatenate([d["ptr"], d["pte"]]).astype(str)
    nom = np.concatenate([d["ntr"], d["nte"]]).astype(str)
    m = y != "none"
    np.savez_compressed(
        salida, X=X[m], E=E[m], y=y[m], participante=part[m], nombre=nom[m]
    )
    return int(m.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="landmarks_cache.npz")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    print(f"{convertir(a.cache, a.salida)} muestras de letras -> {a.salida}")
