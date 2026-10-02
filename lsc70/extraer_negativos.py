"""Extrae más negativos ("gestos que no son seña") y los guarda como landmarks.

    # Desde HaGRID en Hugging Face (la misma fuente del notebook), en streaming:
    python -m lsc70.extraer_negativos --hf cj-mills/hagrid-classification-512p-no-gesture-150k \\
        --clase-origen no_gesture --etiqueta no_gesture_v2 --limite 3000 \\
        --modelo hand_landmarker.task --salida negativos_no_gesture_v2.npz

    # O desde una carpeta de imágenes ya descargada:
    python -m lsc70.extraer_negativos --carpeta HaGRID/no_gesture --etiqueta no_gesture_v2 ...

Usa EXACTAMENTE el mismo pipeline de landmarks que las señas (`landmarks_de_imagen`, con
`num_hands=1`, imagen nativa como en el notebook). El .npz sale en el mismo formato que
`negativos_reposo.npz` (X_neg, clases, clase_por_muestra), así que `exportar_final` lo acepta.

Dos decisiones que conviene entender:
  - NO se filtra por muestra "lo que parece una letra". En `no_gesture` una mano relajada que
    se parece a un puño ES un negativo legítimo; filtrarla escondería justo el problema que se
    quiere arreglar. Solo se excluyen CLASES enteras que son señas (one, palm, three2, two_up).
  - Se guarda con otra etiqueta (`no_gesture_v2`) para no mezclarlo con las 400 del archivo
    viejo; `exportar_final` puede excluir la antigua por nombre y evitar duplicados.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .normalizacion import crear_detector, landmarks_de_imagen

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def imagenes_hf(repo, clase_origen):
    """Itera (imagen BGR) de la clase pedida en un dataset de Hugging Face, en streaming."""
    from datasets import load_dataset, load_dataset_builder

    feats = load_dataset_builder(repo).info.features
    col_label = next(k for k, v in feats.items() if hasattr(v, "names"))
    col_img = next(k for k, v in feats.items() if type(v).__name__ == "Image")
    nombres = list(feats[col_label].names)
    if clase_origen not in nombres:
        raise SystemExit(f"La clase '{clase_origen}' no existe. Clases: {nombres}")
    idx = nombres.index(clase_origen)
    for ej in load_dataset(repo, split="train", streaming=True):
        if ej[col_label] == idx:
            yield cv2.cvtColor(np.array(ej[col_img].convert("RGB")), cv2.COLOR_RGB2BGR)


def imagenes_carpeta(carpeta):
    for r in sorted(Path(carpeta).rglob("*")):
        if r.suffix.lower() in EXT:
            im = cv2.imread(str(r))
            if im is not None:
                yield im


def extraer(imagenes, detector, limite, etiqueta, reporte=500):
    """Landmarks de hasta `limite` imágenes CON mano detectada. Devuelve (X, leidas)."""
    X, leidas, t0 = [], 0, time.time()
    for im in imagenes:
        leidas += 1
        salida = landmarks_de_imagen(detector, im)
        if salida is not None:
            X.append(salida[0])
        if reporte and leidas % reporte == 0:
            print(
                f"  {leidas} leídas | {len(X)} con mano | {time.time() - t0:.0f}s",
                flush=True,
            )
        if len(X) >= limite:
            break
    return np.array(X, np.float32).reshape(-1, 63), leidas


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--hf", help="repositorio de Hugging Face (streaming)")
    fuente.add_argument("--carpeta", help="carpeta con imágenes ya descargadas")
    ap.add_argument(
        "--clase-origen", default="no_gesture", help="clase a leer (solo con --hf)"
    )
    ap.add_argument(
        "--etiqueta", default="no_gesture_v2", help="nombre de clase en el .npz"
    )
    ap.add_argument(
        "--limite", type=int, default=3000, help="máximo de negativos con mano"
    )
    ap.add_argument("--modelo", required=True, help="hand_landmarker.task")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    imgs = imagenes_hf(a.hf, a.clase_origen) if a.hf else imagenes_carpeta(a.carpeta)
    det = crear_detector(a.modelo, num_hands=1)  # igual que para las señas
    X, leidas = extraer(imgs, det, a.limite, a.etiqueta)
    det.close()
    if not len(X):
        raise SystemExit("No se obtuvo ningún negativo con mano. Revisa la fuente.")
    np.savez_compressed(
        a.salida,
        X_neg=X,
        clases=np.array([a.etiqueta]),
        clase_por_muestra=np.array([a.etiqueta] * len(X)),
    )
    descarte = 100 * (1 - len(X) / leidas)
    print(f"\nLeídas {leidas} | con mano {len(X)} | descarte {descarte:.1f} %")
    if len(X) < a.limite:
        print(
            f"La fuente se agotó antes del límite: solo hay {len(X)} (pediste {a.limite})."
        )
    print(f"Guardado {a.salida}")


if __name__ == "__main__":
    main()
