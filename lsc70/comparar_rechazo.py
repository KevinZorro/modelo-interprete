"""Compara dos corridas de `exportar_final --solo-chequeo`, toma a toma y por letra.

    python -m lsc70.comparar_rechazo --antes resultados/A/chequeo_rechazo.json \\
        --despues resultados/C/chequeo_rechazo.json

Las dos corridas usan los MISMOS folds por participante y la misma semilla, así que las tomas
se emparejan por (participante, clase). Solo se comparan las tomas de clases que existen en
ambas (p. ej. las 27 letras si una corrida añade números). El intervalo sale de un bootstrap
que remuestrea PERSONAS, no tomas: las tomas de una persona están correlacionadas.

Por qué no se evalúa el .tflite desplegado: se entrenó con 55 personas y el actual con las 70,
así que cualquier persona que se use para probar el actual ya la vio. Reentrenar ambas
configuraciones con el mismo protocolo es la única comparación sin esa trampa.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RECHAZO = "no_es_seña"


def cargar(ruta):
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {
        (t["participante"], t["real"]): t["pred"]
        for t in d["chequeo"]["registro_tomas"]
    }, d


def bootstrap_diferencia(a, b, personas, n=4000, seed=42):
    """IC95 de mean(b) - mean(a) remuestreando personas. a, b: arrays 0/1 alineados."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(personas)))
    idx = {u: np.where(personas == u)[0] for u in uniq}
    d = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        d[i] = b[s].mean() - a[s].mean()
    return float(b.mean() - a.mean()), *np.percentile(d, [2.5, 97.5])


def comparar(antes, despues):
    """Métricas pareadas sobre las tomas que aparecen en las dos corridas."""
    comunes = sorted(set(antes) & set(despues))
    if not comunes:
        raise SystemExit("Las corridas no comparten ninguna toma.")
    pers = np.array([k[0] for k in comunes])
    real = np.array([k[1] for k in comunes])
    ok_a = np.array([antes[k] == k[1] for k in comunes], float)
    ok_d = np.array([despues[k] == k[1] for k in comunes], float)
    rech_a = np.array([antes[k] == RECHAZO for k in comunes], float)
    rech_d = np.array([despues[k] == RECHAZO for k in comunes], float)
    por_clase = {}
    for c in sorted(set(real)):
        m = real == c
        por_clase[c] = {
            "n": int(m.sum()),
            "antes": float(ok_a[m].mean()),
            "despues": float(ok_d[m].mean()),
            "mejora": int(((ok_d[m] == 1) & (ok_a[m] == 0)).sum()),
            "empeora": int(((ok_d[m] == 0) & (ok_a[m] == 1)).sum()),
        }
    # A dónde van las tomas de letras que el modelo nuevo ya no acierta
    destinos = defaultdict(int)
    destino_por_clase = defaultdict(lambda: defaultdict(int))
    for k in comunes:
        if antes[k] == k[1] and despues[k] != k[1]:
            destinos[despues[k]] += 1
            destino_por_clase[k[1]][despues[k]] += 1
    return {
        "n_tomas": len(comunes),
        "n_personas": len(set(pers)),
        "top1": bootstrap_diferencia(ok_a, ok_d, pers),
        "top1_antes": float(ok_a.mean()),
        "top1_despues": float(ok_d.mean()),
        "falso_rechazo": bootstrap_diferencia(rech_a, rech_d, pers),
        "falso_rechazo_antes": float(rech_a.mean()),
        "falso_rechazo_despues": float(rech_d.mean()),
        "por_clase": por_clase,
        "destino_de_lo_perdido": dict(sorted(destinos.items(), key=lambda kv: -kv[1])),
        "destino_por_clase": {
            c: dict(sorted(d.items(), key=lambda kv: -kv[1]))
            for c, d in destino_por_clase.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--antes", required=True)
    ap.add_argument("--despues", required=True)
    a = ap.parse_args()
    ra, da = cargar(a.antes)
    rd, dd = cargar(a.despues)
    r = comparar(ra, rd)

    def fila(nombre, d):
        print(
            f"{nombre:<24} {100 * d['chequeo']['por_umbral'][0]['falso_rechazo']:>9.1f}% "
            f"{100 * d['chequeo']['por_umbral'][0]['falsa_aceptacion']:>10.1f}% "
            f"{100 * d['chequeo']['por_umbral'][0]['top1_con_rechazo']:>10.1f}%   "
            f"{len(d['clases'])} clases, {sum(1 for _ in d['negativos'])} archivo(s) de negativos"
        )

    print(
        f"{'(argmax, sin umbral)':<24} {'falso rech.':>10} {'falsa acep.':>11} {'top-1':>10}"
    )
    fila("ANTES", da)
    fila("DESPUÉS", dd)
    print(f"\n=== Tomas comunes: {r['n_tomas']} ({r['n_personas']} personas) ===")
    t, lo, hi = r["top1"]
    print(
        f"Top-1: {100 * r['top1_antes']:.1f}% -> {100 * r['top1_despues']:.1f}%   "
        f"Δ {100 * t:+.1f} puntos  IC95 [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
        f"{'(el IC incluye 0: sin diferencia clara)' if lo <= 0 <= hi else '(diferencia clara)'}"
    )
    t, lo, hi = r["falso_rechazo"]
    print(
        f"Señas buenas rechazadas: {100 * r['falso_rechazo_antes']:.1f}% -> "
        f"{100 * r['falso_rechazo_despues']:.1f}%   Δ {100 * t:+.1f}  IC95 [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    )
    print("\nA dónde van las tomas que el modelo ANTES acertaba y DESPUÉS no (top 6):")
    for k, v in list(r["destino_de_lo_perdido"].items())[:6]:
        print(f"  {k:>12}: {v}")
    print(
        f"\n{'clase':>6} {'n':>4} {'antes':>7} {'después':>8} {'Δ':>6}  mejoran/empeoran (tomas)"
    )
    filas = sorted(
        r["por_clase"].items(), key=lambda kv: kv[1]["despues"] - kv[1]["antes"]
    )
    for c, v in filas[:6] + [("...", None)] + filas[-4:]:
        if v is None:
            print("   ...")
            continue
        print(
            f"{c:>6} {v['n']:>4} {100 * v['antes']:>6.0f}% {100 * v['despues']:>7.0f}% "
            f"{100 * (v['despues'] - v['antes']):>+5.0f}  {v['mejora']}/{v['empeora']}"
        )
    print(
        "\nCon ~70 tomas por clase un cambio de una letra de menos de ~10 puntos no se distingue "
        "del ruido; mira el patrón, no cada letra."
    )


if __name__ == "__main__":
    main()
