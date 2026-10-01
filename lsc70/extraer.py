"""Extracción de landmarks de una sección (tareas 3 y 4): MediaPipe + la normalización
del notebook, leyendo del zip sin extraer, con tabla de descarte por clase.

    python -m lsc70.extraer --fuente LSC70AN.zip --nombre an \\
        --modelo hand_landmarker.task --salida landmarks_an.npz

Para comprobar que este pipeline es el mismo del notebook, corre primero ANH y pásale el
cache viejo: compara landmark a landmark las muestras que comparten nombre.

    python -m lsc70.extraer --fuente LSC70ANH.zip --nombre anh ... \\
        --contrastar-cache landmarks_cache.npz
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import Fuente, cargar_estructura, id_toma, parsear
from .normalizacion import PREPARADORES, crear_detector, landmarks_de_imagen

# Descarte medido en ANH por el notebook v8 (solo letras + 'none'). OJO: el 2.8 % que se
# suele citar es SOLO el split de test; train fue 5.2 % y el global 4.6 %.
REFERENCIA_ANH = {"train": 5.2, "test": 2.8, "global": 4.6}


def extraer(fuente, cfg, detector, limite_por_clase=None, reporte=2000):
    prep = PREPARADORES[cfg["preprocesado"]]
    X, E, y, part, nom, frame = [], [], [], [], [], []
    en_disco, retenidas = Counter(), Counter()
    frames_toma_disco, frames_toma_ret = Counter(), Counter()
    ilegibles = 0

    imgs = []
    for n in fuente.imagenes():
        p = parsear(fuente.relativa(n), cfg["idx_participante"], cfg["idx_clase"])
        if p:
            imgs.append((n, *p))
    usadas = Counter()
    t0 = time.time()
    for k, (n, participante, clase, idx) in enumerate(imgs):
        if limite_por_clase and usadas[clase] >= limite_por_clase:
            continue
        usadas[clase] += 1
        en_disco[clase] += 1
        frames_toma_disco[id_toma(participante, clase)] += 1
        im = fuente.leer_imagen(n)
        if im is None:
            ilegibles += 1
            continue
        salida = landmarks_de_imagen(detector, prep(im)[0])
        if salida is None:
            continue  # sin mano (o escala degenerada)
        v, extras, _, _ = salida
        X.append(v)
        E.append(extras)
        y.append(clase)
        part.append(participante)
        # Mismo formato de nombre que el notebook: "{participante}__{clase}__{stem}.jpg"
        nom.append(f"{participante}__{clase}__{Path(n).stem}.jpg")
        frame.append(-1 if idx is None else idx)
        retenidas[clase] += 1
        frames_toma_ret[id_toma(participante, clase)] += 1
        if reporte and (k + 1) % reporte == 0:
            print(f"  {k + 1}/{len(imgs)} | {time.time() - t0:.0f}s", flush=True)

    perdidas = sorted(t for t in frames_toma_disco if frames_toma_ret[t] == 0)
    return (
        {
            "X": np.array(X, np.float32),
            "E": np.array(E, np.float32),
            "y": np.array(y),
            "participante": np.array(part),
            "nombre": np.array(nom),
            "frame": np.array(frame),
            "tomas_perdidas": np.array(perdidas),
            "tomas_en_disco": np.array(sorted(frames_toma_disco)),
        },
        en_disco,
        retenidas,
        ilegibles,
    )


def tabla_descarte(en_disco, retenidas):
    print(f"\n{'clase':>8} {'en disco':>9} {'retenidas':>10} {'descarte %':>11}")
    fila = {}
    for c in sorted(en_disco):
        d, r = en_disco[c], retenidas.get(c, 0)
        fila[c] = {
            "en_disco": d,
            "retenidas": r,
            "descarte_pct": round(100 * (d - r) / d, 2),
        }
        print(f"{c:>8} {d:>9} {r:>10} {fila[c]['descarte_pct']:>10.1f}%")
    td, tr = sum(en_disco.values()), sum(retenidas.values())
    glob = 100 * (td - tr) / td
    print(f"{'TOTAL':>8} {td:>9} {tr:>10} {glob:>10.1f}%")
    print(
        f"\nReferencia ANH (notebook v8, letras + 'none'): train {REFERENCIA_ANH['train']} % | "
        f"test {REFERENCIA_ANH['test']} % | global {REFERENCIA_ANH['global']} %"
    )
    return fila, glob


def contrastar_con_cache(datos, ruta_cache):
    """Compara los landmarks recién extraídos con los del cache del notebook.

    Si el pipeline fuera idéntico, las diferencias serían ruido de punto flotante o de
    versión de MediaPipe. Una diferencia grande delata un paso distinto (orden de la
    normalización, espejo, preprocesado) y hay que parar antes de entrenar nada.
    """
    c = np.load(ruta_cache, allow_pickle=True)
    X_c = np.concatenate([c["Xtr"], c["Xte"]])
    n_c = np.concatenate([c["ntr"], c["nte"]]).astype(str)
    idx_c = {n: i for i, n in enumerate(n_c)}
    comunes = [(i, idx_c[n]) for i, n in enumerate(datos["nombre"]) if n in idx_c]
    if not comunes:
        print(
            "\nContraste con cache: ninguna muestra comparte nombre; nada que comparar."
        )
        return
    a = datos["X"][[i for i, _ in comunes]]
    b = X_c[[j for _, j in comunes]]
    dif = np.abs(a - b).max(axis=1)
    print(
        f"\n=== CONTRASTE CON EL CACHE DEL NOTEBOOK ({len(comunes)} muestras comunes) ==="
    )
    print(
        f"  diferencia máxima por muestra: mediana {np.median(dif):.2e} | p95 {np.percentile(dif, 95):.2e}"
        f" | máx {dif.max():.2e}"
    )
    print(f"  muestras con diferencia < 1e-3: {100 * (dif < 1e-3).mean():.1f} %")
    print(
        "  (esperable: casi todas ~0. Si la mediana es > 1e-2, el pipeline NO es el mismo.)"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--fuente", required=True)
    ap.add_argument("--nombre", required=True, help="'anh' o 'an'")
    ap.add_argument("--modelo", required=True, help="ruta a hand_landmarker.task")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--limite-por-clase", type=int, default=None, help="prueba rápida")
    ap.add_argument("--contrastar-cache", default=None)
    a = ap.parse_args()

    cfg = cargar_estructura(a.nombre)  # se niega si no está confirmada
    fuente = Fuente(a.fuente)
    print(
        f"Sección '{a.nombre}' | preprocesado: {cfg['preprocesado']} | "
        f"{cfg['n_imagenes']} imágenes, {cfg['n_participantes']} participantes"
    )
    detector = crear_detector(
        a.modelo, num_hands=1
    )  # igual que el entrenamiento original
    datos, en_disco, retenidas, ilegibles = extraer(
        fuente, cfg, detector, a.limite_por_clase
    )
    detector.close()
    fila, glob = tabla_descarte(en_disco, retenidas)
    print(
        f"Ilegibles: {ilegibles} | tomas perdidas por completo: {len(datos['tomas_perdidas'])}"
    )

    np.savez_compressed(a.salida, **datos)
    Path(a.salida).with_suffix(".descarte.json").write_text(
        json.dumps(
            {
                "nombre": a.nombre,
                "global_pct": round(glob, 2),
                "por_clase": fila,
                "tomas_perdidas": datos["tomas_perdidas"].tolist(),
                "ilegibles": ilegibles,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Guardado {a.salida} ({datos['X'].shape[0]} muestras)")
    if a.contrastar_cache:
        contrastar_con_cache(datos, a.contrastar_cache)


if __name__ == "__main__":
    main()
