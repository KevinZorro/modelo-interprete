"""¿Los negativos son negativos de verdad, o hay letras disfrazadas entre ellos?

Un negativo legítimo es un gesto que NO es la letra objetivo: rechazarlo es
correcto y subir el umbral para lograrlo es sano. Una "letra disfrazada" es un
gesto de HaGRID que resulta ser la MISMA configuración de mano que una letra LSC
(el caso documentado: peace es V). Usarla como negativo empuja el umbral de esa
letra hacia arriba castigándola por acertar, y el falso rechazo que se ve
después no es un defecto del modelo sino del conjunto de negativos.

La VERIFICACIÓN 2 del notebook ya advertía de esto al elegir los negativos
difíciles. Este script lo comprueba después, sobre los negativos ya exportados.

Uso:
    python -m evaluacion_objetivo.analisis_negativos [--negativos ruta.npz]
"""
import argparse
import json
from collections import Counter

import numpy as np

from .datos import cargar_cache, probabilidades
from .evaluar import RUTA_CACHE, RUTA_CONFIG_SALIDA, RUTA_MODELO
from .metricas import puntaje_objetivo

# Un negativo que el modelo asigna a una letra con esta confianza no es "ruido
# que el modelo confunde": es esa pose. El sondeo de HaGRID del notebook daba
# 0.930 para peace->V y 0.985 para three->W, de ese orden.
UMBRAL_DISFRAZ = 0.80


def falsa_aceptacion_reposo(P, etiquetas, umbrales, k, semilla=42):
    """FA de reposo por letra, por frame y por intento de ~6 frames.

    El umbral se calibra con la FA por FRAME, pero el usuario no hace un frame:
    sostiene la mano medio segundo. Los negativos se agrupan de a 6 al azar como
    proxy de ese intento, igual que hacía el notebook.

    OJO CON ESTA CIFRA: agrupar al azar mezcla 6 imágenes distintas, mientras que
    una mano quieta da 6 frames casi idénticos. Con frames correlacionados, si uno
    pasa el umbral pasan casi todos, así que el valor real de una mano de verdad
    quieta está más cerca de la columna POR FRAME que de esta. Tómala como el
    mejor caso, no como la cifra a reportar.
    """
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(P))
    grupos = [orden[i:i + 6] for i in range(0, len(P) - 5, 6)]

    print(f"\n=== FALSA ACEPTACIÓN DE REPOSO con los umbrales de "
          f"{RUTA_CONFIG_SALIDA.name} ===")
    print(f"{'letra':>5} {'umbral':>7} {'frames':>8} {'por frame':>10} "
          f"{f'por intento (k={k} de 6)':>24}")
    for i, letra in enumerate(etiquetas):
        u = umbrales[letra]
        n_frame = int((P[:, i] >= u).sum())
        n_intento = sum(1 for g in grupos if (P[g, i] >= u).sum() >= k)
        aviso = "  <-- en el tope" if n_frame / len(P) >= 0.0488 else ""
        print(f"{letra:>5} {u:>7.2f} {n_frame:>8} {100*n_frame/len(P):>9.2f}% "
              f"{100*n_intento/len(grupos):>23.1f}%{aviso}")
    print(f"\n({len(grupos)} intentos simulados de 6 frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--negativos", default="negativos_reposo.npz")
    args = ap.parse_args()

    d = np.load(args.negativos, allow_pickle=True)
    X_neg = d["X_neg"]
    P = probabilidades(RUTA_MODELO, X_neg)[:, :27]
    *_, etiquetas = cargar_cache(RUTA_CACHE)

    print(f"Negativos: {len(X_neg)} | clases declaradas: {list(d['clases'])}")
    rechazo = probabilidades(RUTA_MODELO, X_neg).argmax(1) == 27
    print(f"El modelo los manda a 'no_es_seña' en {rechazo.sum()} casos "
          f"({100 * rechazo.mean():.0f} %)\n")

    cfg = json.loads(RUTA_CONFIG_SALIDA.read_text(encoding="utf-8"))
    falsa_aceptacion_reposo(P, etiquetas,
                            {l: v["umbral"] for l, v in cfg["letras"].items()},
                            cfg["k_de_n"])

    print("\n=== POR LETRA: negativos que puntúan alto como esa letra ===")
    print(f"{'letra':>5} {'p>0.5':>7} {'p>0.8':>7} {'p>0.95':>7} {'max':>7}")
    for i, letra in enumerate(etiquetas):
        alto = (P[:, i] > 0.5).sum()
        if alto:
            print(f"{letra:>5} {alto:>7} {(P[:, i] > 0.8).sum():>7} "
                  f"{(P[:, i] > 0.95).sum():>7} {P[:, i].max():>7.3f}")

    if "clase_por_muestra" not in d.files:
        print("\nEl .npz no trae 'clase_por_muestra', así que NO se puede atribuir "
              "cada negativo a su clase de HaGRID y el diagnóstico se queda a "
              "medias: se ve QUÉ letras están afectadas, no POR QUÉ clase.")
        print("Vuelve a exportar con la celda 15-bis actualizada del notebook, "
              "que ya guarda la clase de cada muestra.")
        return

    # Con las clases por muestra sí se puede señalar al culpable concreto.
    clases = d["clase_por_muestra"].astype(str)
    print("\n=== POR CLASE DE HaGRID: letra dominante y confianza ===")
    print(f"{'clase':>18} {'n':>5} {'letra dom.':>11} {'conf. media':>12} "
          f"{f'n con p>{UMBRAL_DISFRAZ}':>14} {'veredicto':>16}")
    sospechosas = {}
    for c, n in sorted(Counter(clases).items()):
        m = clases == c
        medias = P[m].mean(0)
        i_dom = int(medias.argmax())
        n_alto = int((P[m, i_dom] > UMBRAL_DISFRAZ).sum())
        # Se considera disfraz si una parte apreciable de la clase cae en la
        # misma letra con confianza alta, no si hay un caso aislado.
        disfraz = n_alto >= max(3, 0.2 * n)
        if disfraz:
            sospechosas[c] = etiquetas[i_dom]
        print(f"{c:>18} {n:>5} {etiquetas[i_dom]:>11} {medias[i_dom]:>12.3f} "
              f"{n_alto:>14} {'LETRA DISFRAZADA' if disfraz else 'negativo ok':>16}")

    if sospechosas:
        print("\nQuita estas clases de CLASES_NEG_DIFICILES y vuelve a exportar; "
              "mientras estén, el umbral de la letra indicada está inflado:")
        for c, l in sospechosas.items():
            print(f"  {c} -> es la pose de {l}")
    else:
        print("\nNinguna clase se comporta como letra disfrazada.")


if __name__ == "__main__":
    main()
