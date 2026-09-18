"""Descubre clases de equivalencia a partir de la evidencia, no de la intuición.

Dos configuraciones de mano son equivalentes si son LA MISMA pose con distinto
nombre. Eso se distingue de una confusión por error con dos señales:

1. Confusión SIMÉTRICA por toma: a->b y b->a en cantidades parecidas. Una
   confusión por error suele ser asimétrica (la clase mala cae en la buena).
2. CO-ACTIVACIÓN media: probabilidad media que el modelo le da a la clase j
   cuando la real es i. Es más informativa que el argmax, porque mide cuánta
   masa reparte el modelo, no solo quién gana. Si son la misma pose, el modelo
   no puede separarlas y la masa se reparte casi por igual.

Uso:
    python -m evaluacion_objetivo.analisis_equivalencias
"""
import numpy as np

from .datos import datos_evaluacion
from .evaluar import RUTA_CACHE, RUTA_MODELO

# Una equivalencia debe ser fuerte en AMBAS direcciones. Estos mínimos son los
# que separan el único par simétrico real (N/Ñ, 0.300 y 0.305) del resto de
# pares, que no pasan de 0.14 en su mejor dirección.
MIN_COACTIVACION = 0.20
MAX_ASIMETRIA = 2.0     # razón entre la co-activación de ida y la de vuelta


def main():
    d = datos_evaluacion(RUTA_CACHE, RUTA_MODELO)
    ET, y = d["etiquetas"], d["y_toma"]
    K = len(ET)
    PL = d["P_toma"][:, :K]
    pred = PL.argmax(1)

    C = np.zeros((K, K), int)
    for a, b in zip(y, pred):
        C[a, b] += 1

    # Co-activación: fila i = probabilidad media por clase cuando la real es i.
    M = np.stack([PL[y == i].mean(0) for i in range(K)])

    print(f"=== CONFUSIONES POR TOMA (n={len(y)}, "
          f"{len(set(d['part_toma']))} participantes held-out) ===")
    for i in range(K):
        err = sorted(((C[i, j], ET[j]) for j in range(K) if j != i and C[i, j]),
                     reverse=True)
        print(f"{ET[i]:>4} n={C[i].sum():>3} recall={C[i, i]/max(C[i].sum(), 1):.3f} | "
              + ", ".join(f"{c}x{l}" for c, l in err))

    print("\n=== CANDIDATAS A EQUIVALENCIA (ordenadas por co-activación mínima) ===")
    print(f"{'par':>8} {'conf a>b':>9} {'conf b>a':>9} {'coact a>b':>10} "
          f"{'coact b>a':>10} {'veredicto':>14}")
    filas = []
    for i in range(K):
        for j in range(i + 1, K):
            if C[i, j] + C[j, i] == 0 and min(M[i, j], M[j, i]) < 0.05:
                continue
            ida, vuelta = M[i, j], M[j, i]
            minimo = min(ida, vuelta)
            asimetria = max(ida, vuelta) / max(minimo, 1e-9)
            equivale = minimo >= MIN_COACTIVACION and asimetria <= MAX_ASIMETRIA
            filas.append((minimo, ET[i], ET[j], C[i, j], C[j, i], ida, vuelta,
                          "EQUIVALENCIA" if equivale else "error"))
    for minimo, a, b, cab, cba, ida, vuelta, v in sorted(filas, reverse=True)[:12]:
        print(f"{a + '/' + b:>8} {cab:>9} {cba:>9} {ida:>10.3f} {vuelta:>10.3f} {v:>14}")

    print("\nRecordatorio: una equivalencia entre dos letras del vocabulario "
          "implica que NO se pueden enseñar las dos por separado.")


if __name__ == "__main__":
    main()
