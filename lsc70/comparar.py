"""Tarea 7: comparar experimentos (ANH, AN, combinado) con una regla fijada de antemano.

    python -m lsc70.comparar --base resultados/anh_letras_numeros \\
        --otros resultados/an_letras_numeros resultados/comb_letras_numeros

La comparación es PAREADA por toma (participante, clase) y con intervalo bootstrap
remuestreando PARTICIPANTES (no tomas): las seis tomas de una persona están
correlacionadas, así que contarlas como independientes inflaría la confianza.

REGLA DE DECISIÓN (escrita antes de ver resultados de AN):
  1. Métrica principal: top-1 estricto por toma, mismas tomas, mismas personas.
  2. Un modelo "gana" solo si el IC95 bootstrap de la diferencia excluye 0.
  3. Si el IC incluye 0, son equivalentes y se desempata por coincidencia con el
     despliegue: la cámara entrega ESCENAS COMPLETAS, no recortes de la mano (AN se
     parece más). Es un argumento de dominio, no una medición.
  4. Las letras C, K, M, E, H se reportan aparte pero con n=70 por clase un IC95 mide
     ±8-10 puntos: una diferencia de 3 puntos en una sola letra no se puede distinguir
     de ruido. Se mira el patrón, no cada letra.
  5. ANH y AN son las MISMAS personas y las MISMAS tomas: combinar NO es "más datos", es
     una segunda vista de lo mismo (aumento de dominio). No hay participantes nuevos.
"""

import argparse
import json
from pathlib import Path

import numpy as np

CLAVE = ("C", "K", "M", "E", "H")


def cargar(ruta):
    d = np.load(Path(ruta) / "predicciones.npz", allow_pickle=True)
    et = [str(e) for e in d["etiquetas"]]
    return {
        "clave": d["clave"].astype(str),
        "part": d["participante"].astype(str),
        "y": d["y"],
        "ok": d["P"].argmax(1) == d["y"],
        "et": et,
        "metricas": json.loads(
            (Path(ruta) / "metricas.json").read_text(encoding="utf-8")
        ),
    }


def alinear(a, b):
    """Solo las tomas presentes en ambos experimentos, en el mismo orden."""
    comunes = sorted(set(a["clave"]) & set(b["clave"]))
    ia = {k: i for i, k in enumerate(a["clave"])}
    ib = {k: i for i, k in enumerate(b["clave"])}
    return (
        comunes,
        np.array([ia[k] for k in comunes]),
        np.array([ib[k] for k in comunes]),
    )


def bootstrap_participantes(ok_a, ok_b, part, n=4000, seed=42):
    """IC95 de (acierto_b - acierto_a), remuestreando participantes enteros."""
    rng = np.random.default_rng(seed)
    personas = np.array(sorted(set(part)))
    por = {p: np.where(part == p)[0] for p in personas}
    dif = np.empty(n)
    for i in range(n):
        idx = np.concatenate([por[p] for p in rng.choice(personas, len(personas))])
        dif[i] = ok_b[idx].mean() - ok_a[idx].mean()
    return float(ok_b.mean() - ok_a.mean()), *np.percentile(dif, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--base", required=True)
    ap.add_argument("--otros", nargs="+", required=True)
    a = ap.parse_args()
    base = cargar(a.base)

    print(
        f"BASE: {Path(a.base).name} | top-1 {100 * base['metricas']['top1_estricto_ensamble']:.1f} % "
        f"| extremo a extremo {100 * base['metricas']['top1_extremo_a_extremo']:.1f} %\n"
    )
    for ruta in a.otros:
        otro = cargar(ruta)
        claves, ia, ib = alinear(base, otro)
        part = base["part"][ia]
        d, lo, hi = bootstrap_participantes(base["ok"][ia], otro["ok"][ib], part)
        veredicto = (
            "MEJOR que la base"
            if lo > 0
            else "PEOR que la base"
            if hi < 0
            else "equivalente (el IC incluye 0)"
        )
        print(
            f"{Path(ruta).name}: Δ top-1 = {100 * d:+.1f} puntos  IC95 [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            f"  sobre {len(claves)} tomas / {len(set(part))} personas  -> {veredicto}"
        )
        print(
            f"   extremo a extremo: {100 * otro['metricas']['top1_extremo_a_extremo']:.1f} % "
            f"(tomas perdidas: {otro['metricas']['tomas_perdidas']} vs {base['metricas']['tomas_perdidas']})"
        )
        # Letras clave: solo el patrón; con n=70 cada una tiene un IC de ±8-10 puntos.
        celdas = []
        for c in CLAVE:
            if c in base["et"] and c in otro["et"]:
                ka, ko = base["et"].index(c), otro["et"].index(c)
                ma, mb = base["y"][ia] == ka, otro["y"][ib] == ko
                celdas.append(
                    f"{c} {100 * base['ok'][ia][ma].mean():.0f}->{100 * otro['ok'][ib][mb].mean():.0f}"
                )
        print("   recall por letra clave (base->otro, %): " + " | ".join(celdas) + "\n")


if __name__ == "__main__":
    main()
