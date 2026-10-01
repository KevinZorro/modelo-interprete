"""Tarea 2: ¿qué sección tiene más píxeles útiles de mano, ANH o AN?

Mide, para MediaPipe en cada sección, el tamaño de la mano en PÍXELES NATIVOS de la
imagen original (no del lienzo de 480 al que ANH se reescala: ese 2.4x no añade
información). Es lo único que dice si hay más o menos detalle real.

La muestra es PAREADA: las mismas tomas (participante, clase, frame) en ambas secciones,
así que se compara la misma mano en las dos y no dos poblaciones distintas.

    python -m lsc70.medir_mano --anh LSC70ANH.zip --an LSC70AN.zip \\
        --modelo hand_landmarker.task --muestra 600

Métricas por imagen (px nativos):
  - lado: el lado mayor del bounding box de los 21 landmarks.
  - alcance: distancia de la muñeca al punto más lejano (la misma magnitud que la
    `escala` de vector_features, salvo que aquí en 2D y en píxeles).
  - fracción de área: bbox / imagen completa.
Además: tasa de detección y cuántas veces MediaPipe ve DOS manos (en una escena completa
puede elegir la mano que no hace la seña; en un crop de la mano dominante no pasa).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import Fuente, cargar_estructura, parsear
from .normalizacion import PREPARADORES, crear_detector, detectar


def claves(fuente, cfg):
    """{(participante, clase, frame): nombre de entrada}"""
    out = {}
    for n in fuente.imagenes():
        p = parsear(fuente.relativa(n), cfg["idx_participante"], cfg["idx_clase"])
        if p and p[2] is not None:
            out[p] = n
    return out


def medir_imagen(detector, fuente, nombre, cfg):
    """Dict de métricas en px nativos, o {'detectada': False}."""
    im = fuente.leer_imagen(nombre)
    if im is None:
        return {"detectada": False, "ilegible": True}
    h0, w0 = im.shape[:2]
    pasada, (escala, x0, y0) = PREPARADORES[cfg["preprocesado"]](im)
    h, w = pasada.shape[:2]
    res = detectar(detector, pasada)
    if not res.hand_landmarks:
        return {"detectada": False, "tam_original": [w0, h0]}
    pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
    pts = (pts - np.array([x0, y0])) / escala  # del lienzo a px de la imagen original
    ext = pts.max(0) - pts.min(0)
    return {
        "detectada": True,
        "tam_original": [w0, h0],
        "lado_px": float(ext.max()),
        "ancho_px": float(ext[0]),
        "alto_px": float(ext[1]),
        "alcance_px": float(np.linalg.norm(pts - pts[0], axis=1).max()),
        "frac_area": float(ext[0] * ext[1] / (w0 * h0)),
        "dos_manos": len(res.hand_landmarks) > 1,
    }


def resumen(v):
    v = np.array(v, float)
    return {
        "mediana": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
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
    ap.add_argument("--salida", default="resultados/medicion_mano.json")
    a = ap.parse_args()

    sec = {}
    for nombre, ruta in (("anh", a.anh), ("an", a.an)):
        cfg = cargar_estructura(nombre)  # se niega si no está confirmada
        sec[nombre] = (Fuente(ruta), cfg)
    ka, kb = (claves(*sec["anh"]), claves(*sec["an"]))
    comunes = sorted(set(ka) & set(kb))
    print(
        f"Claves (participante, clase, frame): ANH {len(ka)} | AN {len(kb)} | comunes {len(comunes)}"
    )
    if not comunes:
        raise SystemExit(
            "Ninguna imagen coincide entre secciones por nombre; no se puede parear. "
            "Revisa cómo se nombran los archivos de AN."
        )

    # Muestra estratificada por clase para que ninguna letra domine.
    rng = np.random.default_rng(a.semilla)
    por_clase = defaultdict(list)
    for k in comunes:
        por_clase[k[1]].append(k)
    cupo = max(1, a.muestra // len(por_clase))
    elegidas = [
        k
        for c in sorted(por_clase)
        for k in [por_clase[c][i] for i in rng.permutation(len(por_clase[c]))[:cupo]]
    ]
    print(f"Muestra pareada: {len(elegidas)} imágenes ({cupo} por clase)")

    det = crear_detector(a.modelo, num_hands=2)  # 2 para contar manos de más
    filas = []
    for i, k in enumerate(elegidas):
        fila = {"clave": list(k)}
        for nombre in ("anh", "an"):
            fuente, cfg = sec[nombre]
            fila[nombre] = medir_imagen(
                det, fuente, (ka if nombre == "anh" else kb)[k], cfg
            )
        filas.append(fila)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(elegidas)}", flush=True)

    det.close()
    out = {
        "muestra": len(filas),
        "por_seccion": {},
        "pareado": {},
        "por_clase_clave": {},
    }
    print("\n=== TAMAÑO DE LA MANO EN PÍXELES NATIVOS (mediana [p10-p90]) ===")
    for nombre in ("anh", "an"):
        det_ok = [f[nombre] for f in filas if f[nombre].get("detectada")]
        tasa = len(det_ok) / len(filas)
        r = {
            "tasa_deteccion": tasa,
            "tam_original": filas[0][nombre].get("tam_original"),
            "dos_manos_pct": 100 * np.mean([d["dos_manos"] for d in det_ok])
            if det_ok
            else None,
        }
        for m in ("lado_px", "alcance_px", "frac_area"):
            r[m] = resumen([d[m] for d in det_ok])
        out["por_seccion"][nombre] = r
        s = r["lado_px"]
        al = r["alcance_px"]
        fa = r["frac_area"]
        print(
            f"{nombre.upper():>4} | imagen {r['tam_original']} | detección {100 * tasa:.1f} % | "
            f"dos manos {r['dos_manos_pct']:.1f} %"
        )
        print(
            f"      lado del bbox {s['mediana']:.0f} px [{s['p10']:.0f}-{s['p90']:.0f}] | "
            f"alcance {al['mediana']:.0f} px | área {100 * fa['mediana']:.1f} % de la imagen"
        )

    # Comparación pareada: solo imágenes detectadas en AMBAS secciones.
    ambas = [f for f in filas if f["anh"].get("detectada") and f["an"].get("detectada")]
    razon = np.array([f["an"]["lado_px"] / f["anh"]["lado_px"] for f in ambas])
    out["pareado"] = {
        "n": len(ambas),
        "razon_lado_an_sobre_anh": resumen(razon),
        "pct_an_mayor": float(100 * (razon > 1).mean()),
    }
    print(f"\n=== PAREADO ({len(ambas)} imágenes detectadas en ambas) ===")
    print(
        f"Lado de la mano en AN / lado en ANH: mediana {np.median(razon):.2f}x "
        f"[p10 {np.percentile(razon, 10):.2f} - p90 {np.percentile(razon, 90):.2f}]"
    )
    print(
        f"En {100 * (razon > 1).mean():.0f} % de las imágenes la mano tiene MÁS píxeles en AN."
    )
    veredicto = (
        "AN tiene más píxeles útiles de mano"
        if np.median(razon) > 1.1
        else "ANH tiene más píxeles útiles de mano"
        if np.median(razon) < 0.9
        else "ambas secciones tienen un detalle comparable (±10 %)"
    )
    print(f">>> {veredicto}")
    out["veredicto"] = veredicto

    print("\n=== LETRAS CLAVE (mediana del lado, px nativos: ANH -> AN) ===")
    for c in ("C", "K", "M", "E", "H"):
        sub = [f for f in ambas if f["clave"][1] == c]
        if sub:
            print(
                f"  {c}: {np.median([f['anh']['lado_px'] for f in sub]):.0f} -> "
                f"{np.median([f['an']['lado_px'] for f in sub]):.0f}  (n={len(sub)})"
            )
            out["por_clase_clave"][c] = {
                "anh": float(np.median([f["anh"]["lado_px"] for f in sub])),
                "an": float(np.median([f["an"]["lado_px"] for f in sub])),
                "n": len(sub),
            }
    Path(a.salida).parent.mkdir(exist_ok=True)
    Path(a.salida).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nEscrito: {a.salida}")


if __name__ == "__main__":
    main()
