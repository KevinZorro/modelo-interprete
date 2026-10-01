"""CLI de la tarea 1: imprime el árbol y los conteos por clase de una sección."""

import argparse

from .dataset import explorar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fuente", required=True, help="zip o carpeta (p. ej. LSC70AN.zip)"
    )
    ap.add_argument(
        "--nombre", required=True, help="'anh' o 'an' (decide el preprocesado)"
    )
    ap.add_argument("--idx-participante", type=int, default=0)
    ap.add_argument("--idx-clase", type=int, default=1)
    a = ap.parse_args()
    explorar(a.fuente, a.nombre, a.idx_participante, a.idx_clase)


if __name__ == "__main__":
    main()
