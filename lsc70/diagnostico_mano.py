"""Diagnóstico: ¿en AN, MediaPipe elige la mano que hace la seña?

Contexto: en AN (escena completa) MediaPipe ve DOS manos en ~69 % de las imágenes, contra
~3 % en ANH (recorte de la mano dominante). La extracción usa num_hands=1, así que en esas
escenas podría estar quedándose con la mano que NO hace la seña. Eso metería ruido en las
etiquetas y explicaría que AN rinda ~16 puntos peor sin que la resolución tenga nada que
ver. Esto es una hipótesis: este script la mide.

Cómo: ANH es el recorte de la mano dominante de la MISMA toma, así que su vector de 63
features (ya normalizado) es la referencia de "cómo se ve la mano que hace la seña". Para
cada imagen pareada se compara con el vector de la mano que eligió la extracción (num_hands=1)
y con cada mano que ve el detector de dos manos. Si la elegida se parece a la referencia,
es la correcta; si se parece más la otra, la extracción eligió mal.

    python -m lsc70.diagnostico_mano --anh LSC70ANH.zip --an LSC70AN.zip \\
        --modelo hand_landmarker.task --muestra 600

Límite: asume que ANH es de verdad el recorte de la mano que firma. Si el vector de ANH y
el de AN difieren incluso cuando AN ve una sola mano, el ruido de fondo (p95 de esas
distancias) lo dice, y el umbral de "se parece" sale de ahí, no se inventa.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import Fuente, cargar_estructura
from .medir_mano import claves
from .normalizacion import (
    PREPARADORES,
    crear_detector,
    detectar,
    vector_features,
)


def distancia(a, b):
    """Distancia L2 entre dos vectores de 63 features normalizadas."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def vectores_de_manos(res):
    """Un vector (63) por cada mano detectada. Mismo vector_features que el pipeline.

    La lateralidad de cada mano sale de `handedness` en el mismo orden que los landmarks.
    """
    out = []
    for i, lms in enumerate(res.hand_landmarks):
        izq = (
            bool(res.handedness)
            and len(res.handedness) > i
            and res.handedness[i][0].category_name == "Left"
        )
        v, _ = vector_features(lms, izq)
        if v is not None:
            out.append(v)
    return out


def evaluar_imagen(v_ref, v_elegida, v_manos):
    """Compara la mano elegida por el pipeline con la referencia de ANH.

    Devuelve las distancias de la elegida y de todas las manos vistas, y si la elegida es
    la que más se parece a la referencia. Pura y sin MediaPipe, para poder probarla.
    """
    d_elegida = distancia(v_ref, v_elegida)
    d_manos = [distancia(v_ref, v) for v in v_manos]
    return {
        "d_elegida": d_elegida,
        "d_manos": d_manos,
        "n_manos": len(v_manos),
        "elegida_es_la_mejor": d_elegida <= min(d_manos) + 1e-6 if d_manos else True,
    }


def resumir(filas):
    """Tasas globales y por clase. El umbral de 'se parece' sale del ruido medido."""
    una = [f["d_elegida"] for f in filas if f["n_manos"] == 1]
    if len(una) < 20:
        raise SystemExit(
            f"Solo {len(una)} imágenes con una mano en AN: no alcanzan para medir el ruido de fondo."
        )
    # Con una sola mano no hay nada que elegir mal: su distancia a ANH es puro ruido de vista
    # (resolución, recorte, JPEG). El p95 de eso define qué cuenta como "se parece".
    umbral = float(np.percentile(una, 95))
    dos = [f for f in filas if f["n_manos"] >= 2]
    mal = [f for f in dos if f["d_elegida"] > umbral and min(f["d_manos"]) <= umbral]
    por_clase = defaultdict(lambda: [0, 0])
    for f in dos:
        por_clase[f["clave"][1]][1] += 1
        por_clase[f["clave"][1]][0] += f in mal
    return {
        "n_imagenes": len(filas),
        "n_una_mano": len(una),
        "n_dos_manos": len(dos),
        "ruido_una_mano": {
            "mediana": float(np.median(una)),
            "p95_umbral": umbral,
        },
        "dos_manos": {
            "elegida_es_la_mejor_pct": 100
            * float(np.mean([f["elegida_es_la_mejor"] for f in dos]))
            if dos
            else None,
            "mano_equivocada_pct": 100 * len(mal) / len(dos) if dos else None,
            "mano_equivocada_n": len(mal),
        },
        "mano_equivocada_sobre_todas_pct": 100 * len(mal) / len(filas),
        "por_clase_mano_equivocada_pct": {
            c: round(100 * a / b, 1)
            for c, (a, b) in sorted(por_clase.items())
            if b >= 5
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--anh", required=True)
    ap.add_argument("--an", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--muestra", type=int, default=600)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", default="resultados/diagnostico_mano.json")
    a = ap.parse_args()

    cfg_h, cfg_a = cargar_estructura("anh"), cargar_estructura("an")
    f_h, f_a = Fuente(a.anh), Fuente(a.an)
    k_h, k_a = claves(f_h, cfg_h), claves(f_a, cfg_a)
    comunes = sorted(set(k_h) & set(k_a))
    if not comunes:
        raise SystemExit("Ninguna imagen coincide entre secciones; no se puede parear.")
    rng = np.random.default_rng(a.semilla)
    por_clase = defaultdict(list)
    for k in comunes:
        por_clase[k[1]].append(k)
    cupo = max(1, a.muestra // len(por_clase))
    elegidas = [
        por_clase[c][i]
        for c in sorted(por_clase)
        for i in rng.permutation(len(por_clase[c]))[:cupo]
    ]
    print(f"Muestra pareada: {len(elegidas)} imágenes ({cupo} por clase)")

    det1 = crear_detector(a.modelo, num_hands=1)  # lo que usa la extracción
    det2 = crear_detector(a.modelo, num_hands=2)  # para ver todas las manos
    filas = []
    for i, k in enumerate(elegidas):
        ref = detectar(
            det1, PREPARADORES[cfg_h["preprocesado"]](f_h.leer_imagen(k_h[k]))[0]
        )
        v_ref = vectores_de_manos(ref)
        img_an = PREPARADORES[cfg_a["preprocesado"]](f_a.leer_imagen(k_a[k]))[0]
        v_eleg = vectores_de_manos(detectar(det1, img_an))
        v_todas = vectores_de_manos(detectar(det2, img_an))
        if v_ref and v_eleg and v_todas:  # hace falta mano en las tres lecturas
            fila = evaluar_imagen(v_ref[0], v_eleg[0], v_todas)
            fila["clave"] = list(k)
            filas.append(fila)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(elegidas)}", flush=True)
    det1.close()
    det2.close()

    r = resumir(filas)
    r["descartadas_sin_mano"] = len(elegidas) - len(filas)
    d = r["dos_manos"]
    print(
        f"\nImágenes usadas: {r['n_imagenes']} (AN con una mano: {r['n_una_mano']}, con dos: {r['n_dos_manos']})"
    )
    print(
        f"Ruido de fondo (AN con una sola mano vs ANH): mediana {r['ruido_una_mano']['mediana']:.3f}, "
        f"p95 = umbral {r['ruido_una_mano']['p95_umbral']:.3f}"
    )
    if d["mano_equivocada_pct"] is not None:
        print(
            f"Con dos manos: la elegida es la que más se parece a ANH en "
            f"{d['elegida_es_la_mejor_pct']:.1f} %; MANO EQUIVOCADA en {d['mano_equivocada_pct']:.1f} % "
            f"({d['mano_equivocada_n']} imágenes)"
        )
    print(
        f"Sobre todas las imágenes: {r['mano_equivocada_sobre_todas_pct']:.1f} % con la mano equivocada"
    )
    peor = sorted(r["por_clase_mano_equivocada_pct"].items(), key=lambda kv: -kv[1])[:6]
    print(f"Clases con más mano equivocada (% de sus imágenes con dos manos): {peor}")
    Path(a.salida).parent.mkdir(exist_ok=True)
    Path(a.salida).write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nEscrito: {a.salida}")


if __name__ == "__main__":
    main()
