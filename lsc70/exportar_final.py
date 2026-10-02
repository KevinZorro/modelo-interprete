"""Modelo final: 27 letras + números + clase `no_es_seña`, exportado a .tflite.

    python -m lsc70.exportar_final --npz landmarks_anh.npz \\
        --negativos negativos_reposo.npz --salida resultados/modelo_final

Qué hace, en orden:
  1. Chequeo de rechazo con validación cruzada por participante (5 folds): qué parte de las
     tomas bien hechas se rechaza (falso rechazo) y qué parte de los negativos se acepta como
     seña (falsa aceptación). Es la única medida honesta, porque el modelo final ve a todos.
  2. Entrena el modelo final con los 70 participantes (validación por persona para el early
     stopping, como siempre).
  3. Exporta a .tflite y comprueba que da lo mismo que el modelo de Keras.

`entrenar_evaluar` NO guarda modelos: sus resultados son métricas. Este es el único script de
`lsc70` que produce un archivo desplegable.

Límites que conviene tener presentes:
  - Los negativos de HaGRID no traen id de persona: cada uno es su propio grupo, así que la
    falsa aceptación NO es independiente por persona (solo el falso rechazo lo es).
  - Se excluyen las clases de HaGRID que resultaron ser letras (one, palm, three2, two_up).
  - Los números 2 y 3 no existen en el dataset; MIL y MILLON se dejan fuera (sin verificar).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import splits as mod_splits
from .entrenar_evaluar import elegir_clases, entrenar

EXCLUIDAS_POR_DEFECTO = ["one", "palm", "three2", "two_up"]
DETECTOR_MB = (
    7.46  # tamaño del hand_landmarker.task (celda de exportación del notebook)
)
LIMITE_MB = 20  # RNF-03


def cargar_datos(npz, negativos, clases, excluidas):
    d = np.load(npz, allow_pickle=True)
    m = np.isin(d["y"].astype(str), clases)
    idx = {c: i for i, c in enumerate(clases)}
    X = d["X"][m]
    y = np.array([idx[c] for c in d["y"].astype(str)[m]])
    part = d["participante"].astype(str)[m]

    n = np.load(negativos, allow_pickle=True)
    if "clase_por_muestra" not in n.files:
        raise SystemExit(
            "El .npz de negativos no trae clase_por_muestra; no se pueden filtrar."
        )
    quedan = ~np.isin(n["clase_por_muestra"].astype(str), excluidas)
    X_neg = n["X_neg"][quedan]
    print(f"Negativos: {len(n['X_neg'])} -> {len(X_neg)} tras excluir {excluidas}")
    return X, y, part, X_neg


def chequeo_rechazo(X, y, part, X_neg, K, seed, epocas):
    """Validación cruzada del rechazo. Devuelve falso rechazo, falsa aceptación y top-1."""
    from . import entrenar_evaluar as ee

    ee.EPOCHS_MAX = epocas
    fold_de = mod_splits.cargar(sorted(set(part)))["fold_de"]
    f_sena = np.array([fold_de[p] for p in part])
    rng = np.random.default_rng(seed)
    f_neg = rng.permutation(len(X_neg)) % mod_splits.N_FOLDS
    rechazadas = total_tomas = aceptadas_neg = ok_top1 = 0
    for fold in range(mod_splits.N_FOLDS):
        tr, te = f_sena != fold, f_sena == fold
        ntr, nte = f_neg != fold, f_neg == fold
        Xt = np.concatenate([X[tr], X_neg[ntr]])
        yt = np.concatenate([y[tr], np.full(ntr.sum(), K)])
        gt = np.concatenate([part[tr], [f"neg{i}" for i in np.where(ntr)[0]]])
        modelo = entrenar(Xt, yt, gt, K + 1, seed, epochs=epocas)

        P = modelo.predict(X[te], verbose=0)
        claves = {}
        for i, (p, c) in enumerate(zip(part[te], y[te])):
            claves.setdefault((p, int(c)), []).append(i)
        for (_, c), ii in claves.items():
            pred = int(P[ii].mean(0).argmax())
            total_tomas += 1
            rechazadas += pred == K
            ok_top1 += pred == c
        aceptadas_neg += int(
            (modelo.predict(X_neg[nte], verbose=0).argmax(1) != K).sum()
        )
        print(f"  fold {fold + 1}/{mod_splits.N_FOLDS} listo", flush=True)
    return {
        "tomas": total_tomas,
        "falso_rechazo": rechazadas / total_tomas,
        "falsa_aceptacion_frames": aceptadas_neg / len(X_neg),
        "top1_con_rechazo": ok_top1 / total_tomas,
    }


def exportar_tflite(modelo, carpeta, X_ref):
    """Convierte con la misma receta del notebook y verifica la paridad con Keras."""
    import tensorflow as tf

    carpeta = Path(carpeta)
    sm = carpeta / "saved_model"
    modelo.export(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    ruta = carpeta / "signaco_modelo_final.tflite"
    ruta.write_bytes(conv.convert())

    try:  # el intérprete de TF está deprecado; el de LiteRT es el que usa la app
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(ruta))
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    n = min(500, len(X_ref))
    ref = modelo.predict(X_ref[:n], verbose=0)
    out = np.zeros_like(ref)
    for i in range(n):
        it.set_tensor(ent["index"], X_ref[i : i + 1].astype(np.float32))
        it.invoke()
        out[i] = it.get_tensor(sal["index"])[0]
    return ruta, {
        "muestras": n,
        "diferencia_maxima": float(np.abs(ref - out).max()),
        "coincide_argmax_pct": 100 * float((ref.argmax(1) == out.argmax(1)).mean()),
        "forma_entrada": [int(v) for v in ent["shape"]],
        "forma_salida": [int(v) for v in sal["shape"]],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--npz", required=True, help="landmarks_anh.npz")
    ap.add_argument(
        "--negativos", required=True, help="negativos_reposo.npz con clase_por_muestra"
    )
    ap.add_argument("--salida", default="resultados/modelo_final")
    ap.add_argument(
        "--clases", choices=["letras", "letras+numeros"], default="letras+numeros"
    )
    ap.add_argument("--excluir-negativos", nargs="*", default=EXCLUIDAS_POR_DEFECTO)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument(
        "--sin-chequeo", action="store_true", help="salta la validación cruzada"
    )
    a = ap.parse_args()

    presentes = set(np.load(a.npz, allow_pickle=True)["y"].astype(str))
    clases = elegir_clases(a.clases, presentes)
    K = len(clases)
    print(f"Clases ({K}) + no_es_seña (índice {K}): {clases}")
    X, y, part, X_neg = cargar_datos(a.npz, a.negativos, clases, a.excluir_negativos)
    print(
        f"Frames de seña: {len(X)} | negativos: {len(X_neg)} (≈{len(X_neg) / (len(X) / K):.1f}x una clase media)"
    )

    chequeo = None
    if not a.sin_chequeo:
        print("\n=== 1. CHEQUEO DEL RECHAZO (5 folds por participante) ===")
        chequeo = chequeo_rechazo(X, y, part, X_neg, K, a.semilla, a.epocas)
        print(
            f"Falso rechazo (toma bien hecha rechazada): {100 * chequeo['falso_rechazo']:.1f} %\n"
            f"Falsa aceptación (negativo aceptado como seña, por frame): {100 * chequeo['falsa_aceptacion_frames']:.1f} %\n"
            f"Top-1 por toma contando el rechazo como fallo: {100 * chequeo['top1_con_rechazo']:.1f} %"
        )

    print("\n=== 2. ENTRENAMIENTO FINAL (70 participantes) ===")
    import tensorflow as tf

    Xf = np.concatenate([X, X_neg])
    yf = np.concatenate([y, np.full(len(X_neg), K)])
    gf = np.concatenate([part, [f"neg{i}" for i in range(len(X_neg))]])
    modelo = entrenar(Xf, yf, gf, K + 1, a.semilla, epochs=a.epocas)

    print("\n=== 3. EXPORTACIÓN A .tflite ===")
    carpeta = Path(a.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta, paridad = exportar_tflite(modelo, carpeta, Xf)
    mb = ruta.stat().st_size / 1024**2
    print(
        f"{ruta} | {mb:.3f} MB | total con detector {mb + DETECTOR_MB:.2f} MB "
        f"({'CUMPLE' if mb + DETECTOR_MB <= LIMITE_MB else 'NO CUMPLE'} el límite de {LIMITE_MB} MB)"
    )
    print(
        f"Paridad con Keras en {paridad['muestras']} muestras: diferencia máx. "
        f"{paridad['diferencia_maxima']:.2e}, mismo argmax en {paridad['coincide_argmax_pct']:.1f} %"
    )
    if paridad["coincide_argmax_pct"] < 99:
        print("ATENCIÓN: el .tflite no coincide con Keras. No lo despliegues.")

    config = {
        "etiquetas": clases + ["no_es_seña"],
        "indice_rechazo": K,
        "modelo": ruta.name,
        "tamano_mb": round(mb, 4),
        "paridad_tflite": paridad,
        "chequeo_rechazo_cv": chequeo,
        "negativos_excluidos": a.excluir_negativos,
        "verificado_con_camara": False,
        "advertencia": (
            "Modelo entrenado con TODOS los participantes: no tiene held-out propio. Las cifras "
            "de chequeo_rechazo_cv vienen de validación cruzada. Sin probar con cámara y sin "
            "umbrales calibrados (modo objetivo); la decisión es el argmax de 38 clases."
        ),
        "reproducibilidad": {
            "semilla": a.semilla,
            "tensorflow": tf.__version__,
            "datos_sha256": {
                "landmarks": hashlib.sha256(Path(a.npz).read_bytes()).hexdigest()[:16],
                "negativos": hashlib.sha256(Path(a.negativos).read_bytes()).hexdigest()[
                    :16
                ],
            },
        },
    }
    (carpeta / "config_modelo_final.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nEscrito: {carpeta}/ (signaco_modelo_final.tflite, config_modelo_final.json)"
    )


if __name__ == "__main__":
    main()
