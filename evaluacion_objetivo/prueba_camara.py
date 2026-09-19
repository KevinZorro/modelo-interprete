"""Prueba con cámara: genera las planillas y analiza los resultados.

Por qué existe: la prueba que excluyó 8 letras resultó inválida porque quien
probaba ejecutaba mal las señas, y la repetición fue una persona sola sin
protocolo. Una prueba que aguante revisión necesita orden aleatorio, imagen de
referencia a la vista, varias personas y ejecuciones incorrectas a propósito.
Este módulo se encarga de lo mecánico para que nadie improvise.

    python -m evaluacion_objetivo.prueba_camara plantilla --personas 6
    python -m evaluacion_objetivo.prueba_camara analizar pruebas_camara/*.csv

Las métricas y los criterios son los mismos que los del dataset (Wilson, mismos
topes), para que las dos fuentes sean comparables letra a letra.
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .calibracion import clasificar_estado
from .datos import datos_evaluacion
from .evaluar import RUTA_CACHE, RUTA_CONFIG, RUTA_MODELO
from .metricas import wilson

COLUMNAS = ["persona", "bloque", "letra_objetivo", "sena_a_ejecutar",
            "resultado", "intentos", "notas"]
DIR_SALIDA = Path("pruebas_camara")


def confusiones_dirigidas(etiquetas, P, y):
    """Para cada letra, con qué OTRA letra se confunde más.

    Sale de la co-activación medida (probabilidad media de la clase j cuando la
    real es i), no de la intuición: si el modelo va a aceptar una seña
    equivocada, será esa. Es el caso que hay que provocar a propósito.
    """
    M = np.stack([P[y == i, :len(etiquetas)].mean(0) for i in range(len(etiquetas))])
    np.fill_diagonal(M, -1)
    return {letra: etiquetas[int(M[i].argmax())] for i, letra in enumerate(etiquetas)}


def plantilla(args):
    """Una planilla por persona, con las letras en orden aleatorio distinto.

    El orden aleatorio importa: en orden alfabético la persona anticipa la
    siguiente seña y practica mientras espera, lo que mide su memoria, no la app.
    """
    cfg = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    d = datos_evaluacion(RUTA_CACHE, RUTA_MODELO)
    confusion = confusiones_dirigidas(d["etiquetas"], d["P_toma"], d["y_toma"])
    letras = [l for l in d["etiquetas"] if l not in cfg["excluidas_actuales"]["dinamicas"]]

    DIR_SALIDA.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.semilla)
    for n in range(1, args.personas + 1):
        filas = []
        for bloque in ("correcta", "confusion"):
            # Las repeticiones se barajan junto con las letras: si las tres de
            # una letra van seguidas, la persona corrige entre intento e intento
            # y se mide su aprendizaje, no la app.
            orden = [l for l in letras for _ in range(args.intentos)]
            for letra in rng.permutation(orden):
                filas.append({
                    "persona": f"P{n}", "bloque": bloque, "letra_objetivo": letra,
                    "sena_a_ejecutar": letra if bloque == "correcta" else confusion[letra],
                    "resultado": "", "intentos": "", "notas": "",
                })
        ruta = DIR_SALIDA / f"persona_{n}.csv"
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)
        print(f"{ruta}  ({len(filas)} filas)")
    print(f"\nRellena 'resultado' con aceptada/rechazada e 'intentos' con un número.")


def analizar(args):
    """Mismos criterios que el dataset, para que las cifras sean comparables."""
    cfg = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    crit = cfg["criterio_viabilidad"]

    conteo = defaultdict(lambda: {"correcta": [0, 0], "confusion": [0, 0]})
    personas, sin_llenar = set(), 0
    for patron in args.csv:
        for ruta in sorted(glob.glob(patron)):
            with open(ruta, encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    r = fila["resultado"].strip().lower()
                    if r not in ("aceptada", "rechazada"):
                        sin_llenar += 1
                        continue
                    personas.add(fila["persona"])
                    c = conteo[fila["letra_objetivo"]][fila["bloque"]]
                    c[0] += r == "aceptada"
                    c[1] += 1

    if not conteo:
        raise SystemExit("Ninguna fila con resultado. ¿Están llenas las planillas?")
    print(f"Personas: {len(personas)} | filas sin llenar: {sin_llenar}\n")
    print("Recordatorio: los intentos de una misma persona están correlacionados,")
    print("así que estas n son optimistas frente a n personas distintas.\n")
    print(f"{'letra':>5} {'n':>4} {'falso rechazo (IC95)':>24} "
          f"{'acepta la seña mala (IC95)':>28} {'veredicto':>15}")

    resultado = {}
    for letra in sorted(conteo):
        acep_ok, n_ok = conteo[letra]["correcta"]
        acep_mal, n_mal = conteo[letra]["confusion"]
        # Falso rechazo: la hizo bien y no se la aceptaron.
        frr = wilson(n_ok - acep_ok, n_ok)
        # Falsa aceptación dirigida: hizo la seña equivocada y se la dieron por
        # buena. Sin este bloque se mide si la app aprueba, no si enseña.
        fa = wilson(acep_mal, n_mal)
        estado = clasificar_estado({"frr": frr, "fa_cruzada": fa},
                                   crit["frr_max"], crit["fa_cruzada_max"])
        print(f"{letra:>5} {n_ok:>4} {100*frr[0]:>8.1f}% [{100*frr[1]:>5.1f}-{100*frr[2]:>5.1f}] "
              f"{100*fa[0]:>14.1f}% [{100*fa[1]:>5.1f}-{100*fa[2]:>5.1f}] {estado:>15}")
        resultado[letra] = {"frr": frr, "fa_dirigida": fa, "estado": estado,
                            "n_correctas": n_ok, "n_confusiones": n_mal}

    viables = [l for l, v in resultado.items() if v["estado"] == "viable"]
    print(f"\nViables con cámara: {viables or 'ninguna'}")
    print("Estas son las que pueden pasar a verificada_con_camara: true.")
    if args.salida:
        Path(args.salida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"Escrito: {args.salida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plantilla", help="genera una planilla por persona")
    p1.add_argument("--personas", type=int, default=6)
    p1.add_argument("--intentos", type=int, default=3,
                    help="repeticiones por letra; 6 personas x 3 = 18 observaciones, "
                         "el mínimo para que una letra pueda salir viable")
    p1.add_argument("--semilla", type=int, default=42)
    p1.set_defaults(func=plantilla)

    p2 = sub.add_parser("analizar", help="analiza las planillas llenas")
    p2.add_argument("csv", nargs="+")
    p2.add_argument("--salida", default="resultados_camara.json")
    p2.set_defaults(func=analizar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
